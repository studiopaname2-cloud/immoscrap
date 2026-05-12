import os
import io
import gc
import time
import warnings
import urllib.parse

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from shapely import wkt as shapely_wkt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pandas.Int64Index.*')
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

OUTPUT_HTML = os.environ.get('OUTPUT_HTML', 'docs/index.html')
OUTPUT_GEOJSON = os.environ.get('OUTPUT_GEOJSON', 'docs/opportunites.geojson')

# ╔══════════════════════════════════════════════╗
# ║  UNE SEULE LIGNE À MODIFIER                 ║
# ╚══════════════════════════════════════════════╝

CODE_INSEE    = os.environ.get('CODE_INSEE', '93048')
CHEMIN_MAJIC  = os.environ.get('CHEMIN_MAJIC', 'data/majic.parquet')
CHEMIN_LOCAUX = os.environ.get('CHEMIN_LOCAUX', 'data/locaux.parquet')

# Blocage 6 — Normalisation code INSEE
# Certains fichiers parquet stockent '93048', d'autres '093048'
# On normalise toujours en string 5 chiffres
CODE_INSEE = str(CODE_INSEE).strip().zfill(5)
DEPT       = CODE_INSEE[:2]

# Signal Fort — score minimum pour afficher le ping rouge
SIGNAL_FORT_SEUIL = 2

# Dents creuses
HAUT_MAX_DC         = 7.0
ECART_HAUTEUR_MIN_M = 6.0

# Emprise — version corrigée et plus stricte
EMPRISE_VIDE_MAX    = 0.01
EMPRISE_VIDE_M2_MAX = 8.0
EMPRISE_SOUS_MIN    = 0.12
EMPRISE_SOUS_MAX    = 0.25

# Parcelles — seuils de base
SURFACE_MIN_M2 = 120
SURFACE_MAX_M2 = 2500
COMPACITE_MIN  = 0.18

# Blocage 5 — Surface max adaptée par commune
# Saint-Ouen et Aubervilliers ont de grandes parcelles industrielles
# qu'on ne veut pas rater avec le seuil standard de 2500 m²
SURFACE_MAX_PAR_COMMUNE = {
    '93070': 8000,   # Saint-Ouen — grande zone industrielle
    '93001': 8000,   # Aubervilliers — Plaine Saint-Denis
    '93055': 5000,   # Pantin — quelques grandes parcelles
    '93053': 5000,   # Noisy-le-Sec
}
SURFACE_MAX_M2 = SURFACE_MAX_PAR_COMMUNE.get(CODE_INSEE, SURFACE_MAX_M2)
print(f'  Surface max pour {CODE_INSEE} : {SURFACE_MAX_M2} m²')

# Bâtiments / emprise
BATI_MIN_PCI_M2  = 8.0   # ignore les micro-artefacts
BATI_MIN_SOUS_M2 = 20.0
BATI_MAX_SOUS_M2 = 220.0
NB_BAT_MAX_SOUS  = 2

# Biens vacants
ANNEE_DVF_DEBUT      = 2014
ANNEE_DVF_FIN        = 2023
DVF_MIN_ANNEES_ALERTE = 3
EMPRISE_VACANT_MIN   = 0.01
EMPRISE_VACANT_MAX   = 0.35
NATURES_OK = ['Maison','Terrain','Sol',
    'Local industriel. commercial ou assimilé',
    'Dépendance','Bâtiment industriel']

# Couleurs
COUL_DC     = '#E8820C'
COUL_VIDE   = '#2563EB'
COUL_SOUS   = '#7C3AED'
COUL_FRICHE = '#DC2626'
COUL_VACANT = '#92400E'

DOM_PUBLIC = [
    'ETAT','COMMUNE','DEPARTEMENT','REGION','FRANCE','MAIRIE','PREFECTURE',
    'VILLE DE','METROPOLE','SNCF','RATP','OPHLM','OFFICE HLM','HLM',
    'INTERCOMMUNAL','COMMUNAUTE','AGGLOMERATION','SYNDICAT','VOIRIE',
    'DOMAINE PUBLIC','TERRITORIAL','MUNICIPAL','BAILLEUR','HABITAT',
    'LOGEMENT SOCIAL','PARIS HABITAT','ICF','SNI ','SEM ','SPL ',
    'GRAND PARIS','EPA ','EPFIF','APUR','CAISSE DES DEPOTS','CDC ',
    'RFF','VNF','UNIVERSITE','LYCEE','HOPITAL','CHU','APHP'
]

print(f'Commune : {CODE_INSEE} | Département : {DEPT} | Surface max : {SURFACE_MAX_M2} m²')


def resume_annees(annees):
    annees = sorted(int(a) for a in annees if pd.notna(a))
    if not annees:
        return 'aucune'
    if len(annees) == 1:
        return str(annees[0])
    if all(b - a == 1 for a, b in zip(annees, annees[1:])):
        return f'{annees[0]}–{annees[-1]}'
    return ', '.join(str(a) for a in annees)


# Contour commune
def get_commune(code, marge_m=50):
    url = f'https://geo.api.gouv.fr/communes/{code}?format=geojson&geometry=contour'
    for i in range(1, 4):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            gdf = gpd.GeoDataFrame.from_features(
                [{'type':'Feature','geometry':r.json()['geometry'],'properties':{}}],
                crs='EPSG:4326'
            ).to_crs(epsg=2154)
            b = gdf.total_bounds
            return gdf, [b[0]-marge_m, b[1]-marge_m, b[2]+marge_m, b[3]+marge_m]
        except Exception as e:
            print(f'  Tentative {i}/3 : {e}')
            if i < 3:
                time.sleep(3)
    raise Exception(f'Échec pour {code}')


print(f'Contour {CODE_INSEE}...')
commune_gdf, bbox_buf = get_commune(CODE_INSEE)
print(f'✅ Surface : {commune_gdf.geometry.area.sum()/1e6:.2f} km²')

# ══════════════════════════════════════════════════════
# SOURCE 1 — PCI Vecteur (impôts.gouv)
# Tous les bâtiments déclarés → emprise au sol fiable
# ══════════════════════════════════════════════════════
print(f'PCI Vecteur bâti {CODE_INSEE}...')
url_pci = f'https://cadastre.data.gouv.fr/bundler/cadastre-etalab/communes/{CODE_INSEE}/geojson/batiments'

for i in range(1, 4):
    try:
        r = requests.get(url_pci, timeout=120)
        r.raise_for_status()
        bati_pci = gpd.GeoDataFrame.from_features(r.json()['features'], crs='EPSG:4326')
        bati_pci = bati_pci.to_crs(epsg=2154)
        bati_pci['geometry'] = bati_pci.geometry.buffer(0)
        bati_pci = bati_pci[bati_pci.geometry.notna()].copy()
        bati_pci = bati_pci[~bati_pci.geometry.is_empty].copy()
        bati_pci['bat_area'] = bati_pci.geometry.area
        bati_pci = bati_pci[bati_pci['bat_area'] >= BATI_MIN_PCI_M2].copy()
        print(f'✅ PCI : {len(bati_pci)} bâtiments | surface min {bati_pci.bat_area.min():.1f}m² | max {bati_pci.bat_area.max():.1f}m²')
        print(f'   Colonnes : {list(bati_pci.columns)}')
        break
    except Exception as e:
        print(f'  Tentative {i}/3 : {e}')
        if i < 3:
            time.sleep(5)

gc.collect()

# ══════════════════════════════════════════════════════
# SOURCE 2 — BD TOPO IGN WFS
# Bâtiments avec hauteurs → dents creuses uniquement
# ══════════════════════════════════════════════════════
WFS_URL  = 'https://data.geopf.fr/wfs/ows'
LAYER    = 'BDTOPO_V3:batiment'
MAX_REQ  = 1000
bbox_str = f"{bbox_buf[0]:.1f},{bbox_buf[1]:.1f},{bbox_buf[2]:.1f},{bbox_buf[3]:.1f},EPSG:2154"


def wfs_page(offset):
    params = {
        'SERVICE':'WFS','VERSION':'2.0.0','REQUEST':'GetFeature',
        'TYPENAMES':LAYER,'BBOX':bbox_str,'SRSNAME':'EPSG:2154',
        'outputFormat':'application/json','COUNT':MAX_REQ,'STARTINDEX':offset
    }
    for i in range(1, 4):
        try:
            r = requests.get(WFS_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json().get('features', [])
        except Exception as e:
            print(f'    ⚠️ Tentative {i}/3 : {e}')
            if i < 3:
                time.sleep(5)
    raise Exception('WFS échoué')


print('BD TOPO WFS IGN (hauteurs)...')
t0 = time.time()
features, offset = [], 0
while True:
    bloc = wfs_page(offset)
    features.extend(bloc)
    print(f'  offset={offset:5d} → +{len(bloc):4d}  (total : {len(features)})')
    if len(bloc) < MAX_REQ:
        break
    offset += MAX_REQ
    time.sleep(0.3)

bat_topo = gpd.GeoDataFrame.from_features(features, crs='EPSG:2154')
col_h    = next((c for c in bat_topo.columns if c.lower() in ('hauteur','height','z_max','altitude_maximale_toit')), None)
col_u    = next((c for c in bat_topo.columns if c.lower().startswith('usage')), None)
if col_h is None:
    raise Exception(f'Colonne hauteur introuvable : {list(bat_topo.columns)}')

bat_topo = bat_topo.rename(columns={col_h:'hauteur', **({col_u:'usage'} if col_u else {})})
bat_topo['hauteur'] = pd.to_numeric(bat_topo['hauteur'], errors='coerce')

# Clip commune + filtre hauteur valide
bat_topo = gpd.clip(bat_topo, commune_gdf).copy()
bat_dc   = bat_topo[bat_topo['hauteur'] > 0].copy()
if 'usage' in bat_dc.columns:
    usages_ok = ['Résidentiel','Commercial et services','Indifférencié','Industriel']
    bat_dc = bat_dc[bat_dc['usage'].isin(usages_ok) | bat_dc['usage'].isna()].copy()

# Garder uniquement geometry et hauteur — libère RAM
bat_dc = bat_dc[['geometry','hauteur']].copy()
del bat_topo
gc.collect()
print(f'✅ BD TOPO : {len(bat_dc)} bâtiments avec hauteur | {round(time.time()-t0)}s')


# Parcelles cadastrales
# Parcelles cadastrales — avec pagination (l'API plafonne à 1000 par requête)
def charger_parcelles(code, essais=5, delai=15):
    toutes = []
    offset = 0
    limit  = 1000
    while True:
        url = (
            f'https://apicarto.ign.fr/api/cadastre/parcelle'
            f'?code_insee={code}&_limit={limit}&_offset={offset}'
        )
        succes = False
        for i in range(1, essais + 1):
            try:
                print(f'  Essai {i}/{essais} (offset={offset})...')
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                features = r.json().get('features', [])
                toutes.extend(features)
                print(f'  +{len(features)} parcelles (total : {len(toutes)})')
                succes = True
                break
            except Exception as e:
                print(f'  ❌ {str(e)[:80]}')
                if i < essais:
                    print(f'  Retry dans {delai}s...')
                    time.sleep(delai)

        if not succes:
            # On ne plante pas — on travaille avec ce qu'on a
            print(f'  ⚠️ Pagination interrompue à offset={offset} — on continue avec {len(toutes)} parcelles')
            break

        if len(features) < limit:
            break   # Dernière page atteinte
        offset += limit
        time.sleep(1)  # Respecter le rate limit IGN entre pages

    if len(toutes) == 0:
        print(f'  ⚠️ Aucune parcelle récupérée pour {code} — carte vide possible')
        return gpd.GeoDataFrame(columns=['geometry','section','numero','contenance'], crs='EPSG:4326')

    gdf = gpd.GeoDataFrame.from_features(toutes, crs='EPSG:4326')
    print(f'  ✅ {len(gdf)} parcelles au total')
    return gdf


print(f'Parcelles {CODE_INSEE}...')
parcelles = charger_parcelles(CODE_INSEE)
parcelles = parcelles.to_crs(epsg=2154)
parcelles['geometry']   = parcelles.geometry.buffer(0)
parcelles['contenance'] = pd.to_numeric(parcelles['contenance'], errors='coerce')
parcelles['parc_area']  = parcelles.geometry.area
parcelles['cle']        = (
    parcelles['section'].astype(str).str.strip() + '_' +
    parcelles['numero'].astype(str).str.strip().str.zfill(4)
)

# Filtre surface
parcelles = parcelles[
    parcelles['contenance'].between(SURFACE_MIN_M2, SURFACE_MAX_M2)
].copy()

# Filtre voirie par compacité
parcelles['perimetre'] = parcelles.geometry.length
parcelles['compacite'] = (
    4 * np.pi * parcelles['parc_area'] / (parcelles['perimetre'] ** 2)
)
avant     = len(parcelles)
parcelles = parcelles[parcelles['compacite'] >= COMPACITE_MIN].copy()
print(f'  Voirie exclue : {avant} → {len(parcelles)} parcelles')
print(f'✅ {len(parcelles)} parcelles utilisables')

# MAJIC — chargement optimisé (filtre par commune)
print('Chargement MAJIC...')
MAJIC_OK = False
majic_pp = pd.DataFrame(columns=['cle','denomination','siren'])

try:
    schema    = pq.read_schema(CHEMIN_MAJIC)
    cols_m    = schema.names
    c_sec = next((c for c in cols_m if 'section' in c.lower()), None)
    c_num = next((c for c in cols_m if 'numero' in c.lower() and 'section' not in c.lower()), None)
    c_den = next((c for c in cols_m if 'denomination' in c.lower()), None)
    c_sir = next((c for c in cols_m if 'siren' in c.lower()), None)
    c_dep = next((c for c in cols_m if c.lower() in ('ccodep','dep','code_dep')), None)
    c_com = next((c for c in cols_m if c.lower() in ('ccocom','com','code_com')), None)
    print(f'  Colonnes : sec={c_sec} num={c_num} dep={c_dep} com={c_com}')

    commune_3 = CODE_INSEE[2:]
    if c_dep and c_com:
        m = pd.read_parquet(CHEMIN_MAJIC, filters=[(c_dep,'==',DEPT),(c_com,'==',commune_3)])
        print(f'  ✅ Filtre commune : {len(m)} lignes')
    elif c_dep:
        m = pd.read_parquet(CHEMIN_MAJIC, filters=[(c_dep,'==',DEPT)])
        print(f'  ✅ Filtre dept : {len(m)} lignes')
    else:
        m = pd.read_parquet(CHEMIN_MAJIC)
        print(f'  ⚠️ Chargement complet : {len(m)} lignes')

    if c_sec and c_num:
        m['cle'] = (
            m[c_sec].astype(str).str.strip() + '_' +
            m[c_num].astype(str).str.strip().str.zfill(4)
        )
        # On charge toutes les parcelles MAJIC connues
        # Le filtre copropriété (cles_syndic) vient du fichier locaux
        cles_pp   = set(m['cle'].unique())
        cols_keep = ['cle'] + [c for c in [c_den, c_sir] if c]
        majic_pp  = m[cols_keep].drop_duplicates('cle').copy()
        rename_map = {}
        if c_den: rename_map[c_den] = 'denomination'
        if c_sir: rename_map[c_sir] = 'siren'
        majic_pp = majic_pp.rename(columns=rename_map)
        if 'denomination' not in majic_pp.columns: majic_pp['denomination'] = ''
        if 'siren'        not in majic_pp.columns: majic_pp['siren']        = ''
        MAJIC_OK = True
        print(f'✅ MAJIC — {len(cles_pp)} parcelles personnes morales')
    else:
        print('⚠️ Colonnes section/numero introuvables')

    del m
    gc.collect()
except Exception as e:
    print(f'⚠️ MAJIC non chargé : {e}')

# ══════════════════════════════════════════════════════
# LOCAUX — Démembrement (nu-proprio + usufruitier)
# Quand une parcelle a les deux droits simultanément
# c'est souvent une succession bloquée.
# ══════════════════════════════════════════════════════
print('Locaux — démembrement + copropriétés réelles...')
cles_demembrement    = set()
cles_syndic          = set()
cles_multi_sans_synd = set()   # plusieurs entités sur la même parcelle, sans syndic

try:
    fichier_lx = pq.ParquetFile(CHEMIN_LOCAUX)
    lx = fichier_lx.read(
        columns=['code_insee', 'section', 'numero_parcelle', 'code_droit_libelle']
    ).to_pandas()
    lx = lx[lx['code_insee'].astype(str) == CODE_INSEE].copy()
    lx['cle'] = (
        lx['section'].astype(str).str.strip() + '_' +
        lx['numero_parcelle'].astype(str).str.strip().str.zfill(4)
    )

    # Démembrement : nu-proprio + usufruitier sur la même parcelle
    cles_nuprop       = set(lx[lx['code_droit_libelle'] == 'Nu-propriétaire']['cle'])
    cles_usuf         = set(lx[lx['code_droit_libelle'] == 'Usufruitier']['cle'])
    cles_demembrement = cles_nuprop & cles_usuf

    # Vraie copropriété : syndic identifié dans le fichier locaux
    cles_syndic = set(lx[lx['code_droit_libelle'] == 'Syndic de copropriété']['cle'])

    # Multi-entités sans syndic : plusieurs personnes morales sur la même parcelle
    # sans syndic identifié — pas nécessairement une indivision au sens juridique
    nb_prop_lx           = lx.groupby('cle').size().reset_index(name='nb_prop')
    cles_multi           = set(nb_prop_lx[nb_prop_lx['nb_prop'] > 1]['cle'])
    cles_multi_sans_synd = cles_multi - cles_syndic

    print(f'✅ Locaux — {len(cles_demembrement)} démembrements | '
          f'{len(cles_syndic)} syndics | '
          f'{len(cles_multi_sans_synd)} multi-entités sans syndic')
    del lx
    gc.collect()
except Exception as e:
    print(f'⚠️ Locaux non chargé : {e}')


# =========================================================
# EMPRISE — VERSION CORRIGÉE
# 1) calcul rapide par centroïdes
# 2) recalcul précis par intersection pour les faibles emprises
# =========================================================
print('Calcul emprise au sol corrigée...')

# Calcul rapide par centroïdes
pci_pts = bati_pci.copy()
pci_pts['geometry'] = pci_pts.geometry.centroid

joined = gpd.sjoin(
    pci_pts[['geometry','bat_area']],
    parcelles[['geometry','cle']],
    how='inner', predicate='within'
).drop(columns='index_right', errors='ignore')

emprise_fast = joined.groupby('cle').agg(
    emprise_m2=('bat_area','sum'),
    nb_bat_pci=('bat_area','count')
).reset_index()

del pci_pts, joined
gc.collect()

# Hauteurs BD TOPO par parcelle
bat_pts = bat_dc.copy()
bat_pts['geometry'] = bat_pts.geometry.centroid
h_join = gpd.sjoin(
    bat_pts[['geometry','hauteur']],
    parcelles[['geometry','cle']],
    how='inner', predicate='within'
).drop(columns='index_right', errors='ignore')
h_agg = h_join.groupby('cle').agg(
    haut_max_bat=('hauteur','max'),
    haut_min_bat=('hauteur','min')
).reset_index()

del bat_pts, h_join
gc.collect()

parcelles = parcelles.merge(emprise_fast, on='cle', how='left')
parcelles = parcelles.merge(h_agg, on='cle', how='left')
for col in ['emprise_m2','haut_max_bat','haut_min_bat']:
    parcelles[col] = parcelles[col].fillna(0)
parcelles['nb_bat_pci'] = parcelles['nb_bat_pci'].fillna(0).astype(int)
parcelles['emprise_ratio'] = parcelles['emprise_m2'] / parcelles['parc_area'].replace(0, np.nan)
parcelles['emprise_ratio'] = parcelles['emprise_ratio'].fillna(0)

del emprise_fast, h_agg
gc.collect()

# Correction précise pour les parcelles à faible emprise / candidates
suspects = parcelles[
    (parcelles['emprise_ratio'] <= 0.30) &
    (parcelles['contenance'] >= SURFACE_MIN_M2)
][['cle','geometry']].copy()

if len(suspects) > 0:
    print(f'  Correction précise sur {len(suspects)} parcelles suspectes...')
    inter = gpd.overlay(
        bati_pci[['geometry']],
        suspects[['cle','geometry']],
        how='intersection'
    )

    if len(inter) > 0:
        inter['inter_area'] = inter.geometry.area
        inter = inter[inter['inter_area'] >= BATI_MIN_PCI_M2].copy()

        emprise_precise = inter.groupby('cle').agg(
            emprise_m2_precise=('inter_area','sum'),
            nb_bat_pci_precise=('inter_area','count')
        ).reset_index()

        parcelles = parcelles.merge(emprise_precise, on='cle', how='left')
        mask_precise = parcelles['emprise_m2_precise'].notna()
        parcelles.loc[mask_precise, 'emprise_m2'] = parcelles.loc[mask_precise, 'emprise_m2_precise']
        parcelles.loc[mask_precise, 'nb_bat_pci'] = parcelles.loc[mask_precise, 'nb_bat_pci_precise']
        parcelles['nb_bat_pci'] = parcelles['nb_bat_pci'].fillna(0).astype(int)
        parcelles['emprise_ratio'] = parcelles['emprise_m2'] / parcelles['parc_area'].replace(0, np.nan)
        parcelles['emprise_ratio'] = parcelles['emprise_ratio'].fillna(0)
        parcelles = parcelles.drop(columns=['emprise_m2_precise','nb_bat_pci_precise'], errors='ignore')

    del inter
    gc.collect()

del suspects
gc.collect()

print('✅ Emprise calculée via hybride centroïde + intersection')
print(f'   Terrains vides stricts : {((parcelles["emprise_ratio"] <= EMPRISE_VIDE_MAX) & (parcelles["emprise_m2"] <= EMPRISE_VIDE_M2_MAX)).sum()}')
print(f'   Sous-exploités filtrés : {parcelles["emprise_ratio"].between(EMPRISE_SOUS_MIN, EMPRISE_SOUS_MAX).sum()}')


# Cartofriches CEREMA — avec cache disque
# Sur 9 communes dans le même workflow : téléchargé une seule fois
print('Cartofriches...')
CACHE_CARTO = '/tmp/cartofriches_cache.csv'
url_carto   = (
    'https://static.data.gouv.fr/resources/sites-references-dans-cartofriches'
    '/20250422-125540/friches-standard.csv'
)
try:
    if os.path.exists(CACHE_CARTO):
        print('  Cache trouvé — lecture locale')
        carto_all = pd.read_csv(CACHE_CARTO, sep=';', low_memory=False)
    else:
        print('  Téléchargement Cartofriches...')
        r = requests.get(url_carto, timeout=120)
        r.raise_for_status()
        with open(CACHE_CARTO, 'w', encoding='utf-8') as f:
            f.write(r.text)
        carto_all = pd.read_csv(io.StringIO(r.text), sep=';', low_memory=False)
        print(f'  Cache écrit : {CACHE_CARTO}')
except Exception as e:
    print(f'  ⚠️ Cartofriches inaccessible : {e} — friches désactivées')
    carto_all = pd.DataFrame()

if len(carto_all) == 0:
    col_insee = col_nom = col_adr = col_surf = col_type = col_statut = col_url = col_prop = None
    friches = pd.DataFrame()
    gdf_friches = gpd.GeoDataFrame()
    print('⚠️  Cartofriches non disponible — calque friches vide')
else:
    col_insee  = next((c for c in carto_all.columns if 'insee'   in c.lower()), None)
    col_nom    = next((c for c in carto_all.columns if 'nom'     in c.lower() and 'site' in c.lower()), None)
    col_adr    = next((c for c in carto_all.columns if 'adresse' in c.lower()), None)
    col_surf   = next((c for c in carto_all.columns if 'surface' in c.lower()), None)
    col_type   = next((c for c in carto_all.columns if 'type'    in c.lower() and 'site' in c.lower()), None)
    col_statut = next((c for c in carto_all.columns if 'statut'  in c.lower()), None)
    col_url    = next((c for c in carto_all.columns if 'url'     in c.lower()), None)
    col_prop   = next((c for c in carto_all.columns if 'proprio' in c.lower() or 'proprietaire' in c.lower()), None)

    friches = carto_all[carto_all[col_insee].astype(str) == CODE_INSEE].copy() if col_insee else pd.DataFrame()
    print(f'  {len(friches)} friche(s) à {CODE_INSEE}')

    # Géolocalisation
    col_pt  = next((c for c in friches.columns if friches[c].astype(str).str.contains('POINT', na=False).any()), None) if len(friches) > 0 else None
    col_lat = next((c for c in friches.columns if c.lower() in ('lat','latitude')), None)  if len(friches) > 0 else None
    col_lon = next((c for c in friches.columns if c.lower() in ('lon','lng','longitude')), None) if len(friches) > 0 else None

    if len(friches) == 0:
        gdf_friches = gpd.GeoDataFrame()
        print('  ℹ️  Aucune friche pour cette commune')
    elif col_pt:
        friches['geometry'] = friches[col_pt].apply(
            lambda x: shapely_wkt.loads(str(x)) if pd.notna(x) and 'POINT' in str(x) else None)
        gdf_friches = gpd.GeoDataFrame(friches.dropna(subset=['geometry']), crs='EPSG:4326')
    elif col_lat and col_lon:
        friches['_lat'] = pd.to_numeric(friches[col_lat], errors='coerce')
        friches['_lon'] = pd.to_numeric(friches[col_lon], errors='coerce')
        friches = friches.dropna(subset=['_lat','_lon'])
        gdf_friches = gpd.GeoDataFrame(
            friches,
            geometry=gpd.points_from_xy(friches['_lon'], friches['_lat']),
            crs='EPSG:4326'
        )
    else:
        gdf_friches = gpd.GeoDataFrame()
        print('  ⚠️  Pas de coordonnées dans Cartofriches')

    if len(gdf_friches) > 0:
        print(f'✅ {len(gdf_friches)} friches géolocalisées')

# ══════════════════════════════════════════════════════
# DVF 2014–2023
# ══════════════════════════════════════════════════════
print(f'DVF {ANNEE_DVF_DEBUT}–{ANNEE_DVF_FIN}...')
dvf_frames = []
dvf_stats = []

for annee in range(ANNEE_DVF_DEBUT, ANNEE_DVF_FIN + 1):
    url = f'https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/communes/{DEPT}/{CODE_INSEE}.csv'
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            cols_keep = [c for c in [
                'id_parcelle','date_mutation','nature_mutation',
                'nature_culture','valeur_fonciere','surface_reelle_bati'
            ] if c in df.columns]
            df = df[cols_keep].copy()
            if 'nature_culture' in df.columns:
                df = df[
                    df['nature_culture'].isin(NATURES_OK) |
                    df['nature_culture'].isna()
                ].copy()
            if 'nature_mutation' in df.columns:
                df = df[
                    df['nature_mutation'].isin([
                        'Vente','Adjudication','Expropriation',
                        'Donation','Echange',"Vente en l'état futur d'achèvement"
                    ])
                ].copy()
            dvf_frames.append(df)
            dvf_stats.append({'annee': annee, 'status': 200, 'rows': len(df)})
            print(f'  {annee} ✅ {len(df)} mutations')
        elif r.status_code == 404:
            dvf_stats.append({'annee': annee, 'status': 404, 'rows': 0})
            print(f'  {annee} ℹ️  Fichier absent')
        elif r.status_code == 429:
            print(f'  {annee} ⚠️  Rate limit — attente 30s...')
            time.sleep(30)
            r2 = requests.get(url, timeout=30)
            if r2.status_code == 200:
                df = pd.read_csv(io.StringIO(r2.text), low_memory=False)
                cols_keep = [c for c in [
                    'id_parcelle','date_mutation','nature_mutation',
                    'nature_culture','valeur_fonciere','surface_reelle_bati'
                ] if c in df.columns]
                df = df[cols_keep].copy()
                if 'nature_culture' in df.columns:
                    df = df[
                        df['nature_culture'].isin(NATURES_OK) |
                        df['nature_culture'].isna()
                    ].copy()
                if 'nature_mutation' in df.columns:
                    df = df[
                        df['nature_mutation'].isin([
                            'Vente','Adjudication','Expropriation',
                            'Donation','Echange',"Vente en l'état futur d'achèvement"
                        ])
                    ].copy()
                dvf_frames.append(df)
                dvf_stats.append({'annee': annee, 'status': 200, 'rows': len(df)})
                print(f'  {annee} ✅ Réessai ok — {len(df)} mutations')
            else:
                dvf_stats.append({'annee': annee, 'status': r2.status_code, 'rows': 0})
                print(f'  {annee} ❌ {r2.status_code} après réessai')
        else:
            dvf_stats.append({'annee': annee, 'status': r.status_code, 'rows': 0})
            print(f'  {annee} ❌ {r.status_code}')
    except Exception as e:
        dvf_stats.append({'annee': annee, 'status': 'ERR', 'rows': 0})
        print(f'  {annee} ❌ {str(e)[:50]}')
    time.sleep(1)

annees_dvf_dispo = [int(x['annee']) for x in dvf_stats if x['status'] == 200]
annees_dvf_ko = [f"{x['annee']} ({x['status']})" for x in dvf_stats if x['status'] != 200]
periode_dvf_label = resume_annees(annees_dvf_dispo)
dvf_alerte_active = len(annees_dvf_dispo) >= DVF_MIN_ANNEES_ALERTE

if dvf_frames:
    dvf = pd.concat(dvf_frames, ignore_index=True)
    dvf['section_dvf'] = dvf['id_parcelle'].astype(str).str[8:10].str.strip()
    dvf['numero_dvf']  = dvf['id_parcelle'].astype(str).str[10:].str.strip().str.zfill(4)
    dvf['cle']         = dvf['section_dvf'] + '_' + dvf['numero_dvf']
    dvf['date_mutation'] = pd.to_datetime(dvf['date_mutation'], errors='coerce')

    cles_avec_mutation = set(dvf['cle'].unique())

    dvf_last = dvf.sort_values('date_mutation').groupby('cle').last().reset_index()
    dvf_last = dvf_last.rename(columns={
        'date_mutation': 'derniere_mutation_date',
        'nature_mutation': 'derniere_mutation_type',
        'nature_culture': 'nature_bien',
        'valeur_fonciere': 'dernier_prix'
    })

    print(f'\n✅ {len(cles_avec_mutation)} parcelles avec mutation sur période DVF disponible ({periode_dvf_label})')
    print(f'   Types : {dvf["nature_mutation"].value_counts().to_dict()}')
else:
    cles_avec_mutation = set()
    dvf_last = pd.DataFrame()
    print('⚠️ DVF non chargé')

print(f"DVF années OK : {periode_dvf_label}")
if annees_dvf_ko:
    print(f"DVF années manquantes / erreurs : {', '.join(annees_dvf_ko)}")
else:
    print(f'DVF : toutes les années {ANNEE_DVF_DEBUT}–{ANNEE_DVF_FIN} sont revenues pour cette commune')

if dvf_alerte_active:
    print(f"✅ Alerte DVF active sur la période disponible ({periode_dvf_label})")
else:
    print(f"⚠️ Alerte DVF désactivée : seulement {len(annees_dvf_dispo)} année(s) disponible(s)")

DATE_ACHAT_RECENTE = pd.Timestamp.today().normalize() - pd.DateOffset(years=5)
if len(dvf_last) > 0:
    cles_achat_recent = set(
        dvf_last[dvf_last['derniere_mutation_date'] >= DATE_ACHAT_RECENTE]['cle']
    )
    print(f"✅ {len(cles_achat_recent)} parcelles achetées il y a moins de 5 ans exclues (selon DVF disponible)")
else:
    cles_achat_recent = set()


# Analyse : 4 catégories
print('Analyse...')

# Pleine propriété
if MAJIC_OK:
    parc_pp = parcelles.merge(
        majic_pp[['cle','denomination','siren']], on='cle', how='left'
    )
    parc_pp['denomination'] = parc_pp['denomination'].fillna('Particulier')
    parc_pp['siren']        = parc_pp['siren'].fillna('')
    parc_pp = parc_pp[~parc_pp['cle'].isin(cles_syndic)].copy()
    masque = parc_pp['denomination'].str.upper().apply(
        lambda x: any(d in x for d in DOM_PUBLIC)
    )
    parc_pp = parc_pp[~masque].copy()
else:
    parc_pp = parcelles.copy()
    parc_pp['denomination'] = 'Particulier'
    parc_pp['siren']        = ''

if len(dvf_last) > 0:
    parc_pp = parc_pp.merge(
        dvf_last[['cle', 'derniere_mutation_date', 'derniere_mutation_type']],
        on='cle', how='left'
    )
else:
    parc_pp['derniere_mutation_date'] = pd.NaT
    parc_pp['derniere_mutation_type'] = ''

parc_pp = parc_pp[~parc_pp['cle'].isin(cles_achat_recent)].copy()
print(f'  {len(parc_pp)} parcelles pleine propriété privée')

# Catégorie 1 : Terrains vides stricts
vides = parc_pp[
    (parc_pp['contenance'] >= SURFACE_MIN_M2) &
    (parc_pp['emprise_ratio'] <= EMPRISE_VIDE_MAX) &
    (parc_pp['emprise_m2'] <= EMPRISE_VIDE_M2_MAX)
].copy()
vides['categorie']       = 'Terrain vide'
vides['hauteur']         = 0.0
vides['ecart_max']       = 0.0
vides['haut_voisin_max'] = 0.0
print(f'  🔵 {len(vides)} terrains vides (stricts)')

# Catégorie 2 : Sous-exploités potentiels — version durcie
sous = parc_pp[
    (parc_pp['contenance'] >= 180) &
    (parc_pp['emprise_ratio'].between(EMPRISE_SOUS_MIN, EMPRISE_SOUS_MAX)) &
    (parc_pp['emprise_m2'] >= BATI_MIN_SOUS_M2) &
    (parc_pp['emprise_m2'] <= BATI_MAX_SOUS_M2) &
    (parc_pp['nb_bat_pci'] <= NB_BAT_MAX_SOUS)
].copy()
sous['categorie']       = 'Sous-exploité'
sous['hauteur']         = sous['haut_min_bat']
sous['ecart_max']       = 0.0
sous['haut_voisin_max'] = 0.0
print(f'  🟣 {len(sous)} sous-exploités (filtrés)')

# Catégorie 3 : Dents creuses RDC/R+1
bat_bas = bat_dc[bat_dc['hauteur'] <= HAUT_MAX_DC].copy().reset_index(drop=True)
bat_ref = bat_dc.copy().reset_index(drop=True)
bat_buf = bat_bas.copy()
bat_buf['geometry'] = bat_buf.geometry.buffer(0.5)

joined = gpd.sjoin(
    bat_buf[['geometry','hauteur']].rename(columns={'hauteur':'haut_bas'}),
    bat_ref[['geometry','hauteur']].rename(columns={'hauteur':'haut_voisin'}),
    how='inner', predicate='intersects'
)
joined  = joined[joined.index != joined['index_right']].copy()
joined['ecart'] = joined['haut_voisin'] - joined['haut_bas']
vh      = joined[joined['ecart'] >= ECART_HAUTEUR_MIN_M]
stats   = vh.groupby(vh.index).agg(
    ecart_max=('ecart','max'),
    haut_voisin_max=('haut_voisin','max')
)
dc_bat  = bat_bas.join(stats, how='inner')
dc_pts  = dc_bat.copy()
dc_pts['geometry'] = dc_pts.geometry.centroid

dc_parc = gpd.sjoin(
    dc_pts[['geometry','hauteur','ecart_max','haut_voisin_max']],
    parc_pp[['geometry','cle','section','numero','contenance','parc_area',
              'emprise_ratio','compacite','denomination','siren',
              'derniere_mutation_date','derniere_mutation_type']],
    how='inner', predicate='within'
).drop(columns='index_right', errors='ignore')
dc_parc = dc_parc.sort_values('ecart_max', ascending=False).drop_duplicates(subset='cle', keep='first').copy()
dc_parc['categorie'] = 'Dent creuse'
print(f'  🟠 {len(dc_parc)} dents creuses RDC/R+1')

# Géométrie parcelle pour affichage
parc_geom = parcelles[['cle','geometry']].rename(columns={'geometry':'geom_parcelle'})
dc_parc = dc_parc.merge(parc_geom, on='cle', how='left')
vides   = vides.merge(parc_geom, on='cle', how='left')
sous    = sous.merge(parc_geom, on='cle', how='left')

# ── Candidats Signal Fort vacants ───────────────
# On calcule les candidats vacants pour le Signal Fort
# mais on ne les affiche PAS comme calque séparé
candidats = parcelles.copy()
# Exclure uniquement les vraies copropriétés (syndic identifié)
candidats = candidats[~candidats['cle'].isin(cles_syndic)].copy()
candidats = candidats[~candidats['cle'].isin(cles_achat_recent)].copy()
if cles_avec_mutation:
    candidats = candidats[~candidats['cle'].isin(cles_avec_mutation)].copy()
candidats = candidats[
    candidats['emprise_ratio'].between(EMPRISE_VACANT_MIN, EMPRISE_VACANT_MAX)
].copy()
if len(majic_pp) > 0:
    candidats = candidats.merge(majic_pp[['cle','denomination','siren']], on='cle', how='left')
else:
    candidats['denomination'] = ''
    candidats['siren'] = ''
candidats['denomination'] = candidats['denomination'].fillna('Particulier')
candidats['siren']        = candidats['siren'].fillna('')
masque_pub = candidats['denomination'].str.upper().apply(
    lambda x: any(d in x for d in DOM_PUBLIC)
)
candidats    = candidats[~masque_pub].copy()
parc_vacants = candidats.merge(parc_geom, on='cle', how='left').copy()
if len(dvf_last) > 0:
    parc_vacants = parc_vacants.merge(
        dvf_last[['cle', 'derniere_mutation_date', 'derniere_mutation_type']],
        on='cle', how='left'
    )
else:
    parc_vacants['derniere_mutation_date'] = pd.NaT
    parc_vacants['derniere_mutation_type'] = ''
parc_vacants['categorie'] = 'Bien vacant'
print(f'  🟤 {len(parc_vacants)} candidats vacants (Signal Fort uniquement)')

# Total = uniquement les catégories affichées sur la carte
# parc_vacants n'est PAS affiché — seuls les vacants Signal Fort le seront
total_affiche = len(dc_parc) + len(vides) + len(sous) + len(gdf_friches)
print(f'\n✅ Total affiché : {total_affiche} opportunités (hors vacants Signal Fort)')

# ══════════════════════════════════════════════════════
# SIGNAL FORT — score par cumul de signaux
#
# +1  Aucune vente DVF sur période fiable
# +1  Démembrement (nu-proprio + usufruitier)
# +1  Répertorié dans Cartofriches
# +1  Pleine propriété société (SIREN connu)
#
# Seuil : SIGNAL_FORT_SEUIL signaux → ping rouge
# ══════════════════════════════════════════════════════
print('Calcul Signal Fort...')

# Récupérer la clé cadastrale des friches via jointure spatiale
cles_friches = set()
if len(gdf_friches) > 0:
    try:
        friches_2154 = gdf_friches.to_crs(epsg=2154).copy()
        friches_2154['geometry'] = friches_2154.geometry.centroid
        joined_f = gpd.sjoin(
            friches_2154[['geometry']],
            parcelles[['geometry','cle']],
            how='inner', predicate='within'
        )
        cles_friches = set(joined_f['cle'].dropna())
        print(f'  {len(cles_friches)} parcelles identifiées comme friches')
    except Exception as e:
        print(f'  ⚠️ Jointure friches/parcelles : {e}')

# Les vacants sont déjà filtrés par DVF=0 → ce signal ne les discrimine pas
# On ne le compte donc que pour les autres catégories
def score_signal_fort(cle, siren='', compter_dvf=True):
    score = 0
    if compter_dvf and dvf_alerte_active and cle not in cles_avec_mutation:
        score += 1
    if cle in cles_demembrement:
        score += 1  # Succession potentiellement bloquée
    if cle in cles_friches:
        score += 1  # Répertoriée comme friche
    if siren and str(siren) not in ('', 'nan'):
        score += 1  # Propriétaire société identifiée
    return score


def ajouter_score(df, compter_dvf=True):
    df = df.copy()
    df['signal_fort_score'] = df.apply(
        lambda r: score_signal_fort(
            r.get('cle', ''), r.get('siren', ''), compter_dvf
        ), axis=1
    )
    df['signal_fort'] = df['signal_fort_score'] >= SIGNAL_FORT_SEUIL
    return df


dc_parc      = ajouter_score(dc_parc,      compter_dvf=True)
vides        = ajouter_score(vides,         compter_dvf=True)
sous         = ajouter_score(sous,          compter_dvf=True)
parc_vacants = ajouter_score(parc_vacants,  compter_dvf=False)  # déjà filtré
vac_fort     = parc_vacants[parc_vacants['signal_fort_score'] >= SIGNAL_FORT_SEUIL].copy()


def geocoder_df(df):
    if len(df) == 0:
        df = df.copy()
        df['adresse'] = []
        return df

    pts = df.copy()
    if pts.crs and pts.crs.to_epsg() != 4326:
        pts = pts.to_crs(epsg=4326)
    centroids = pts.geometry.centroid

    # Batch — on envoie toutes les coordonnées en une seule requête CSV
    # au lieu d'un appel par ligne → 10-20x plus rapide
    lignes = ['longitude,latitude,idx']
    for i, geom in enumerate(centroids):
        lignes.append(f'{round(geom.x,6)},{round(geom.y,6)},{i}')
    csv_data = '\n'.join(lignes)

    try:
        r = requests.post(
            'https://api-adresse.data.gouv.fr/reverse/csv/',
            files={'data': ('coords.csv', csv_data.encode(), 'text/csv')},
            timeout=120
        )
        r.raise_for_status()
        result = pd.read_csv(io.StringIO(r.text))
        col_label = next(
            (c for c in result.columns if 'result_label' in c.lower() or c.lower() == 'label'),
            None
        )
        if col_label and 'idx' in result.columns:
            result = result.sort_values('idx')
            adresses = result[col_label].fillna('Adresse inconnue').tolist()
        else:
            adresses = ['Adresse inconnue'] * len(df)
    except Exception as e:
        print(f'  ⚠️ Géocodage batch échoué : {e} — adresses inconnues')
        adresses = ['Adresse inconnue'] * len(df)

    df = df.copy()
    df['adresse'] = adresses
    return df


print('Géocodage...')
dc_parc   = geocoder_df(dc_parc)
vides     = geocoder_df(vides)
sous      = geocoder_df(sous)
vac_fort  = geocoder_df(vac_fort)
if len(gdf_friches) > 0:
    gdf_friches = geocoder_df(gdf_friches)
print('✅ Adresses récupérées')

# Associer chaque friche à sa parcelle cadastrale
friche_parc_map = {}
if len(gdf_friches) > 0:
    try:
        fr_pts = gdf_friches.to_crs(epsg=2154).copy()
        fr_pts['friche_idx'] = list(fr_pts.index)
        fr_pts['geometry'] = fr_pts.geometry.centroid
        fr_join = gpd.sjoin(
            fr_pts[['friche_idx', 'geometry']],
            parc_pp[['cle', 'section', 'numero', 'contenance', 'denomination', 'siren',
                     'derniere_mutation_date', 'derniere_mutation_type', 'geometry']],
            how='left', predicate='within'
        ).drop(columns='index_right', errors='ignore')
        parc_geom_fri = parc_pp[['cle', 'geometry']].rename(columns={'geometry': 'geom_parcelle'})
        fr_join = fr_join.merge(parc_geom_fri, on='cle', how='left')
        friche_parc_map = fr_join.set_index('friche_idx').to_dict('index')
    except Exception as e:
        print(f'  ⚠️ Association friches/parcelles échouée : {e}')

if len(gdf_friches) > 0:
    gdf_friches = gdf_friches.copy()
    gdf_friches['cle'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('cle', ''))
    gdf_friches['section'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('section', ''))
    gdf_friches['numero'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('numero', ''))
    gdf_friches['contenance'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('contenance', np.nan))
    gdf_friches['denomination'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('denomination', ''))
    gdf_friches['siren'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('siren', ''))
    gdf_friches['derniere_mutation_date'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('derniere_mutation_date', pd.NaT))
    gdf_friches['derniere_mutation_type'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('derniere_mutation_type', ''))
    gdf_friches['geom_parcelle'] = gdf_friches.index.map(lambda i: friche_parc_map.get(i, {}).get('geom_parcelle'))
    gdf_friches = gdf_friches[~gdf_friches['cle'].isin(cles_achat_recent)].copy()
    friche_parc_map = {k: v for k, v in friche_parc_map.items() if k in set(gdf_friches.index)}
    gdf_friches = ajouter_score(gdf_friches, compter_dvf=True)
else:
    gdf_friches['signal_fort_score'] = pd.Series(dtype='int64')
    gdf_friches['signal_fort'] = pd.Series(dtype='bool')

nb_fort = int(sum([
    dc_parc['signal_fort'].sum(),
    vides['signal_fort'].sum(),
    sous['signal_fort'].sum(),
    vac_fort['signal_fort'].sum(),
    gdf_friches['signal_fort'].sum() if 'signal_fort' in gdf_friches.columns else 0
]))
print(f'✅ {nb_fort} biens avec Signal Fort (≥{SIGNAL_FORT_SEUIL} signaux)')


def niveau(h):
    if h <= 0:
        return 'Vide'
    if h <= 3.5:
        return 'RDC'
    if h <= 7:
        return 'R+1'
    return 'R+2+'


def type_prop(p, s):
    if p in ('Particulier','Inconnu','N/A',''):
        return 'Particulier'
    return 'Société'


def format_derniere_mutation(date_val):
    if pd.isna(date_val) or str(date_val) in ('', 'nan', 'NaT'):
        if dvf_alerte_active:
            return f'Aucune vente DVF sur période disponible ({periode_dvf_label})'
        if annees_dvf_dispo:
            return f'DVF partiel disponible ({periode_dvf_label})'
        return 'DVF indisponible'
    try:
        return pd.to_datetime(date_val).strftime('%d/%m/%Y')
    except Exception:
        return str(date_val)


centre = parcelles.to_crs(epsg=4326).geometry.centroid.unary_union.centroid
carte  = folium.Map(location=[centre.y, centre.x], zoom_start=15)

folium.TileLayer(
    tiles=('https://wmts.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile'
           '&VERSION=1.0.0&LAYER=CADASTRALPARCELS.PARCELLAIRE_EXPRESS'
           '&STYLE=normal&FORMAT=image/png&TILEMATRIXSET=PM'
           '&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}'),
    attr='© IGN', name='Parcelles IGN', overlay=True, control=True, opacity=0.55
).add_to(carte)

# CSS animation ping rouge — injecté une seule fois dans la page
carte.get_root().html.add_child(folium.Element("""
<style>
@keyframes ping {
  0%   { transform: scale(1);   opacity: 1; }
  80%  { transform: scale(2.5); opacity: 0; }
  100% { transform: scale(2.5); opacity: 0; }
}
.ping-rouge {
  width: 14px; height: 14px;
  background: #DC2626;
  border-radius: 50%;
  border: 2px solid white;
  box-shadow: 0 0 0 0 rgba(220,38,38,0.6);
  animation: ping 1.4s ease-out infinite;
  position: absolute;
  top: -18px; left: -7px;
  pointer-events: none;
  z-index: 9999;
}
</style>
"""))

fg_dc  = folium.FeatureGroup(name=f'🟠 Dents creuses ({len(dc_parc)})', show=True)
fg_vid = folium.FeatureGroup(name=f'🔵 Terrains vides stricts ({len(vides)})', show=True)
fg_sou = folium.FeatureGroup(name=f'🟣 Sous-exploités filtrés ({len(sous)})', show=True)
fg_fri = folium.FeatureGroup(name=f'🔴 Friches ({len(gdf_friches)})', show=True)
fg_fort = folium.FeatureGroup(name=f'🚨 Signal Fort ({nb_fort})', show=True)


def ajouter(fg, gp, lat, lon, coul, col_f, icone, html, tip, sec, num, fort=False, fg_fort_ref=None):
    geom_wgs = gpd.GeoSeries([gp], crs='EPSG:2154').to_crs('EPSG:4326').iloc[0]
    folium.GeoJson(
        geom_wgs.__geo_interface__,
        style_function=lambda x, c=coul: {'color': c, 'weight': 2, 'fillOpacity': 0.4}
    ).add_to(fg)
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(html, max_width=320),
        tooltip=tip,
        icon=folium.Icon(color=col_f, icon=icone, prefix='fa')
    ).add_to(fg)
    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(
            html=(f'<div style="font-family:Arial;font-size:9px;font-weight:bold;'
                  f'color:#1e293b;background:rgba(255,255,255,0.85);'
                  f'padding:1px 4px;border-radius:3px;border:1px solid {coul};'
                  f'white-space:nowrap;margin-top:22px;margin-left:8px;'
                  f'pointer-events:none">{sec} {num}</div>'),
            icon_size=(60,18), icon_anchor=(0,0)
        )
    ).add_to(fg)
    # Fix 1 — fg_fort passé explicitement, pas pris depuis le contexte global
    # Fix 5 — comparaison explicite au lieu de bool() fragile
    if fort is True and fg_fort_ref is not None:
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html='<div class="ping-rouge"></div>',
                icon_size=(14, 14),
                icon_anchor=(7, 28)
            )
        ).add_to(fg_fort_ref)


def get_gp_latlon(row):
    gp = row.get('geom_parcelle') or row['geometry']
    gw = gpd.GeoSeries([gp], crs='EPSG:2154').to_crs('EPSG:4326').iloc[0]
    c  = gw.centroid
    return gp, round(c.y, 6), round(c.x, 6)


def popup_base(rang, cat, coul, adr, sec, num, surf, emp, prop, sir):
    ae = urllib.parse.quote(adr)
    gm = f'https://www.google.com/maps/search/?api=1&query={ae}'
    ge = f'https://earth.google.com/web/search/{ae}'
    sir_h = f"<b>SIREN :</b> {sir}<br>" if sir and sir not in ('','nan') else ''
    return gm, ge, sir_h


# 🟠 Dents creuses
print(f'Dents creuses ({len(dc_parc)})...')
for rang, (_, row) in enumerate(dc_parc.iterrows(), 1):
    try:
        sec = str(row.get('section','')).strip()
        num = str(row.get('numero','')).strip()
        surf = int(row['contenance']) if pd.notna(row.get('contenance')) else '?'
        adr = str(row.get('adresse','') or 'Adresse inconnue')
        prop = str(row.get('denomination','Particulier'))
        sir = str(row.get('siren','') or '')
        haut = round(float(row.get('hauteur',0) or 0), 1)
        vois = round(float(row.get('haut_voisin_max',0) or 0), 1)
        ecar = round(float(row.get('ecart_max',0) or 0), 1)
        emp = round(float(row.get('emprise_ratio',0) or 0) * 100, 1)
        mut = format_derniere_mutation(row.get('derniere_mutation_date'))
        gp, lat, lon = get_gp_latlon(row)
        gm, ge, sir_h = popup_base(rang,'DC',COUL_DC,adr,sec,num,surf,emp,prop,sir)
        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_DC}'>#{rang} Dent creuse | {niveau(haut)}</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Parcelle :</b> {sec} n°{num} | <b>Surface :</b> {surf} m²<br>"
            f"<b>Emprise :</b> {emp}% | <b>Haut :</b> {haut}m | <b>Voisin :</b> {vois}m | <b style='color:{COUL_DC}'>Écart : {ecar}m</b>"
            f"<hr style='margin:5px 0'><b>Proprio :</b> {prop} ({type_prop(prop,sir)})<br>{sir_h}"
            f"<b>Dernière mutation :</b> {mut}<br>"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        ajouter(fg_dc,gp,lat,lon,COUL_DC,'orange','arrow-up',html,f'#{rang} Dent creuse | {adr}',sec,num,
                fort=row.get('signal_fort_score',0)>=SIGNAL_FORT_SEUIL, fg_fort_ref=fg_fort)
    except Exception as e:
        print(f'  ⚠️ DC {rang}: {e}')

# 🔵 Terrains vides
print(f'Terrains vides ({len(vides)})...')
for rang, (_, row) in enumerate(vides.iterrows(), 1):
    try:
        sec = str(row.get('section','')).strip()
        num = str(row.get('numero','')).strip()
        surf = int(row['contenance']) if pd.notna(row.get('contenance')) else '?'
        adr = str(row.get('adresse','') or 'Adresse inconnue')
        prop = str(row.get('denomination','Particulier'))
        sir = str(row.get('siren','') or '')
        emp = round(float(row.get('emprise_ratio',0) or 0) * 100, 1)
        emp_m2 = round(float(row.get('emprise_m2',0) or 0), 1)
        mut = format_derniere_mutation(row.get('derniere_mutation_date'))
        gp, lat, lon = get_gp_latlon(row)
        gm, ge, sir_h = popup_base(rang,'Vide',COUL_VIDE,adr,sec,num,surf,emp,prop,sir)
        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_VIDE}'>#{rang} Terrain vide strict</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Parcelle :</b> {sec} n°{num} | <b>Surface :</b> {surf} m²<br>"
            f"<b>Emprise PCI :</b> {emp}% | <b>Bâti détecté :</b> {emp_m2} m² max"
            f"<hr style='margin:5px 0'><b>Proprio :</b> {prop} ({type_prop(prop,sir)})<br>{sir_h}"
            f"<b>Dernière mutation :</b> {mut}<br>"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        ajouter(fg_vid,gp,lat,lon,COUL_VIDE,'blue','tint',html,f'#{rang} Vide | {adr}',sec,num,
                fort=row.get('signal_fort_score',0)>=SIGNAL_FORT_SEUIL, fg_fort_ref=fg_fort)
    except Exception as e:
        print(f'  ⚠️ Vide {rang}: {e}')

# 🟣 Sous-exploités
print(f'Sous-exploités ({len(sous)})...')
for rang, (_, row) in enumerate(sous.iterrows(), 1):
    try:
        sec = str(row.get('section','')).strip()
        num = str(row.get('numero','')).strip()
        surf = int(row['contenance']) if pd.notna(row.get('contenance')) else '?'
        adr = str(row.get('adresse','') or 'Adresse inconnue')
        prop = str(row.get('denomination','Particulier'))
        sir = str(row.get('siren','') or '')
        haut = round(float(row.get('haut_min_bat',0) or 0), 1)
        emp = round(float(row.get('emprise_ratio',0) or 0) * 100, 1)
        emp_m2 = round(float(row.get('emprise_m2',0) or 0), 1)
        nb_bat = int(row.get('nb_bat_pci', 0) or 0)
        mut = format_derniere_mutation(row.get('derniere_mutation_date'))
        gp, lat, lon = get_gp_latlon(row)
        gm, ge, sir_h = popup_base(rang,'Sous',COUL_SOUS,adr,sec,num,surf,emp,prop,sir)
        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_SOUS}'>#{rang} Sous-exploité filtré</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Parcelle :</b> {sec} n°{num} | <b>Surface :</b> {surf} m²<br>"
            f"<b>Emprise :</b> {emp}% | <b>Surface bâtie :</b> {emp_m2} m² | <b>Nb bâtis :</b> {nb_bat}<br>"
            f"<b>Haut. min :</b> {haut}m"
            f"<hr style='margin:5px 0'><b>Proprio :</b> {prop} ({type_prop(prop,sir)})<br>{sir_h}"
            f"<b>Dernière mutation :</b> {mut}<br>"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        ajouter(fg_sou,gp,lat,lon,COUL_SOUS,'purple','building',html,f'#{rang} Sous-exp. | {adr}',sec,num,
                fort=row.get('signal_fort_score',0)>=SIGNAL_FORT_SEUIL, fg_fort_ref=fg_fort)
    except Exception as e:
        print(f'  ⚠️ Sous {rang}: {e}')

# 🔴 Friches
print(f'Friches ({len(gdf_friches)})...')
for rang, (idx, row) in enumerate(gdf_friches.iterrows(), 1):
    try:
        nom = str(row.get(col_nom,'') or f'Friche #{rang}')
        adr = str(row.get(col_adr,'') or row.get('adresse','') or 'Adresse inconnue')
        surf = str(row.get(col_surf,'?') or '?')
        typ = str(row.get(col_type,'?') or '?')
        stat = str(row.get(col_statut,'?') or '?')
        url_f = str(row.get(col_url,'') or '') if col_url else ''
        prop_f = str(row.get(col_prop,'?') or '?') if col_prop else '?'
        geom = row.geometry
        c = geom.centroid if geom.geom_type != 'Point' else geom
        lat, lon = round(c.y, 6), round(c.x, 6)
        ae = urllib.parse.quote(adr)
        gm = f'https://www.google.com/maps/search/?api=1&query={ae}'
        ge = f'https://earth.google.com/web/search/{ae}'

        infos_parc = friche_parc_map.get(idx, {})
        gp = infos_parc.get('geom_parcelle')
        sec = str(infos_parc.get('section', '') or '').strip()
        num = str(infos_parc.get('numero', '') or '').strip()
        surf_parc = infos_parc.get('contenance')
        surf_parc = int(surf_parc) if pd.notna(surf_parc) else '?'
        prop_parc = str(infos_parc.get('denomination', '') or '').strip()
        mut_parc = format_derniere_mutation(infos_parc.get('derniere_mutation_date'))

        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_FRICHE}'>#{rang} Friche répertoriée</b><br>"
            f"<b>{nom}</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"{f'<b>Parcelle :</b> {sec} n°{num} | <b>Surface parcelle :</b> {surf_parc} m²<br>' if sec or num else ''}"
            f"<b>Type :</b> {typ} | <b>Statut :</b> {stat}<br>"
            f"<b>Surface site :</b> {surf} m² | <b>Proprio site :</b> {prop_f}<br>"
            f"{f'<b>Proprio parcelle :</b> {prop_parc}<br>' if prop_parc else ''}"
            f"{f'<b>Dernière mutation :</b> {mut_parc}<br>' if sec or num else ''}"
            f"<b>Source :</b> Cartofriches CEREMA"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a>"
            f"{'<br><br><a href=' + repr(url_f) + ' target=_blank style=font-size:11px;color:#666>Fiche →</a>' if url_f else ''}"
            f"</div>"
        )

        if gp is not None:
            geom_wgs = gpd.GeoSeries([gp], crs='EPSG:2154').to_crs('EPSG:4326').iloc[0]
            parc_cent = geom_wgs.centroid
            poly_lat, poly_lon = round(parc_cent.y, 6), round(parc_cent.x, 6)
            folium.GeoJson(
                geom_wgs.__geo_interface__,
                style_function=lambda x: {'color': COUL_FRICHE, 'weight': 2, 'fillOpacity': 0.4}
            ).add_to(fg_fri)
            if sec or num:
                folium.Marker(
                    [poly_lat, poly_lon],
                    icon=folium.DivIcon(
                        html=(f'<div style="font-family:Arial;font-size:9px;font-weight:bold;'
                              f'color:#1e293b;background:rgba(255,255,255,0.85);'
                              f'padding:1px 4px;border-radius:3px;border:1px solid {COUL_FRICHE};'
                              f'white-space:nowrap;margin-top:22px;margin-left:8px;'
                              f'pointer-events:none">{sec} {num}</div>'),
                        icon_size=(60,18), icon_anchor=(0,0)
                    )
                ).add_to(fg_fri)
            lat, lon = poly_lat, poly_lon

        folium.Marker(
            [lat, lon], popup=folium.Popup(html, max_width=320),
            tooltip=f'#{rang} Friche | {nom}',
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(fg_fri)

        if geom.geom_type != 'Point':
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda x: {'color': COUL_FRICHE, 'weight': 2, 'fillOpacity': 0.15}
            ).add_to(fg_fri)

        if row.get('signal_fort_score', 0) >= SIGNAL_FORT_SEUIL:
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html='<div class="ping-rouge"></div>',
                    icon_size=(14, 14),
                    icon_anchor=(7, 28)
                )
            ).add_to(fg_fort)
    except Exception as e:
        print(f'  ⚠️ Friche {rang}: {e}')

# 🚨 Vacants Signal Fort — affichés directement dans fg_fort
print(f'Vacants Signal Fort ({len(vac_fort)})...')
for rang, (_, row) in enumerate(vac_fort.iterrows(), 1):
    try:
        sec   = str(row.get('section','')).strip()
        num   = str(row.get('numero','')).strip()
        surf  = int(row['contenance']) if pd.notna(row.get('contenance')) else '?'
        adr   = str(row.get('adresse','') or 'Adresse inconnue')
        prop  = str(row.get('denomination','Particulier'))
        sir   = str(row.get('siren','') or '')
        emp   = round(float(row.get('emprise_ratio',0) or 0)*100, 1)
        score = int(row.get('signal_fort_score', 0))
        cle   = row.get('cle','')
        mut   = format_derniere_mutation(row.get('derniere_mutation_date'))
        gp, lat, lon = get_gp_latlon(row)
        ae = urllib.parse.quote(adr)
        gm = f'https://www.google.com/maps/search/?api=1&query={ae}'
        ge = f'https://earth.google.com/web/search/{ae}'
        sir_h = f"<b>SIREN :</b> {sir}<br>" if sir and sir not in ('','nan') else ''

        signaux = []
        if cle in cles_demembrement: signaux.append('✓ Démembrement (succession)')
        if cle in cles_friches:      signaux.append('✓ Friche répertoriée')
        if sir and sir not in ('','nan'): signaux.append('✓ Société identifiée')
        signaux.append(f'✓ Aucune vente DVF sur période disponible ({periode_dvf_label})' if dvf_alerte_active else '✓ Aucune vente DVF observée sur période partielle')
        # Indivision — info seulement, pas dans le score
        if cle in cles_multi_sans_synd:
            signaux.append('ℹ️ Plusieurs entités sur la parcelle (sans syndic)')

        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:280px;line-height:1.9'>"
            f"<b style='font-size:15px;color:#DC2626'>🚨 Signal Fort ({score} signaux)</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Parcelle :</b> {sec} n°{num} | <b>Surface :</b> {surf} m²<br>"
            f"<b>Emprise bâtie :</b> {emp}%"
            f"<hr style='margin:5px 0'><b>Proprio :</b> {prop} ({type_prop(prop,sir)})<br>{sir_h}"
            f"<b>Dernière mutation :</b> {mut}<br>"
            f"<hr style='margin:5px 0'>{'<br>'.join(signaux)}"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        geom_wgs = gpd.GeoSeries([gp], crs='EPSG:2154').to_crs('EPSG:4326').iloc[0]
        folium.GeoJson(
            geom_wgs.__geo_interface__,
            style_function=lambda x: {'color':'#DC2626','weight':2,'fillOpacity':0.4}
        ).add_to(fg_fort)
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(html, max_width=330),
            tooltip=f'🚨 Signal Fort | {adr} | {score} signaux',
            icon=folium.Icon(color='red', icon='exclamation', prefix='fa')
        ).add_to(fg_fort)
    except Exception as e:
        print(f'  ⚠️ Vacant fort {rang}: {e}')

for fg in [fg_dc, fg_vid, fg_sou, fg_fri, fg_fort]:
    fg.add_to(carte)
folium.LayerControl(collapsed=False, position='topright').add_to(carte)

total = len(dc_parc)+len(vides)+len(sous)+len(gdf_friches)
carte.get_root().html.add_child(folium.Element(
    f"<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
    f"background:white;padding:14px 18px;border-radius:10px;"
    f"box-shadow:0 2px 10px rgba(0,0,0,.2);font-family:Arial;font-size:12px;min-width:240px'>"
    f"<b>{CODE_INSEE} — {total} opportunités</b><br>"
    f"<i style='color:#888;font-size:10px'>Calques cochables → haut droite</i><br><br>"
    f"<span style='color:{COUL_DC}'>&#9679;</span> Dents creuses {len(dc_parc)}<br>"
    f"<span style='color:{COUL_VIDE}'>&#9679;</span> Terrains vides stricts {len(vides)}<br>"
    f"<span style='color:{COUL_SOUS}'>&#9679;</span> Sous-exploités filtrés {len(sous)}<br>"
    f"<span style='color:{COUL_FRICHE}'>&#9679;</span> Friches {len(gdf_friches)}<br>"
    f"<hr style='margin:6px 0'>"
    f"<span style='color:#DC2626'>&#9679;</span> <b>Signal Fort</b> {nb_fort}"
    f"<i style='color:#999;font-size:10px;display:block;margin-top:3px'>"
    f"≥{SIGNAL_FORT_SEUIL} signaux — ping rouge + marqueurs dédiés</i>"
    f"</div>"
))

print(f'✅ Carte affichée — {total} opportunités + {nb_fort} Signal Fort')
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
carte.save(OUTPUT_HTML)
print(f'✅ HTML généré : {OUTPUT_HTML}')

try:
    exports = []

    def _prep_export(df, categorie):
        if len(df) == 0:
            return None
        df = df.copy()
        if 'geom_parcelle' in df.columns:
            df['geometry'] = df['geom_parcelle']
        if getattr(df, 'crs', None) is None:
            df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:2154')
        if df.crs.to_epsg() != 4326:
            df = df.to_crs(epsg=4326)
        df['categorie_export'] = categorie
        return df

    for _df, _cat in [
        (dc_parc,    'Dent creuse'),
        (vides,      'Terrain vide'),
        (sous,       'Sous-exploité'),
        (gdf_friches,'Friche'),
        (vac_fort,   'Signal Fort vacant'),
    ]:
        e = _prep_export(_df, _cat)
        if e is not None:
            exports.append(e)

    if exports:
        all_gdf = pd.concat(exports, ignore_index=True)
        all_gdf = gpd.GeoDataFrame(all_gdf, geometry='geometry', crs='EPSG:4326')
        os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
        all_gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
        print(f'✅ GeoJSON exporté : {OUTPUT_GEOJSON}')
except Exception as e:
    print(f'⚠️ Export GeoJSON ignoré : {e}')
