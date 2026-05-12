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

warnings.filterwarnings('ignore')

OUTPUT_HTML = os.environ.get('OUTPUT_HTML', 'docs/index.html')
OUTPUT_GEOJSON = os.environ.get('OUTPUT_GEOJSON', 'docs/opportunites.geojson')

# ╔══════════════════════════════════════════════╗
# ║  UNE SEULE LIGNE À MODIFIER                 ║
# ╚══════════════════════════════════════════════╝

CODE_INSEE = os.environ.get('CODE_INSEE', '93048')
CHEMIN_MAJIC = os.environ.get('CHEMIN_MAJIC', 'data/majic.parquet')

# Dents creuses
HAUT_MAX_DC         = 7.0
ECART_HAUTEUR_MIN_M = 6.0

# Emprise — version corrigée et plus stricte
EMPRISE_VIDE_MAX    = 0.01   # 1%
EMPRISE_VIDE_M2_MAX = 8.0    # max 8 m² bâtis pour rester "terrain vide"
EMPRISE_SOUS_MIN    = 0.12   # 12%
EMPRISE_SOUS_MAX    = 0.25   # 25%

# Parcelles
SURFACE_MIN_M2 = 120
SURFACE_MAX_M2 = 2500
COMPACITE_MIN  = 0.18

# Bâtiments / emprise
BATI_MIN_PCI_M2  = 8.0   # ignore les micro-artefacts
BATI_MIN_SOUS_M2 = 20.0
BATI_MAX_SOUS_M2 = 220.0
NB_BAT_MAX_SOUS  = 2

# Biens vacants
ANNEE_DVF_DEBUT    = 2014
ANNEE_DVF_FIN      = 2023
EMPRISE_VACANT_MIN = 0.01
EMPRISE_VACANT_MAX = 0.35
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

DEPT = CODE_INSEE[:2]
print(f'Commune : {CODE_INSEE}')


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
def charger_parcelles(code, essais=5, delai=15):
    url = f'https://apicarto.ign.fr/api/cadastre/parcelle?code_insee={code}'
    for i in range(1, essais+1):
        try:
            print(f'  Essai {i}/{essais}...')
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            gdf = gpd.GeoDataFrame.from_features(r.json()['features'], crs='EPSG:4326')
            if len(gdf) == 0:
                raise ValueError('0 parcelles')
            print(f'  ✅ {len(gdf)} parcelles')
            return gdf
        except Exception as e:
            print(f'  ❌ {str(e)[:80]}')
            if i < essais:
                print(f'  Retry dans {delai}s...')
                time.sleep(delai)
    raise Exception('Cadastre inaccessible')


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
MAJIC_OK   = False
majic_pp   = pd.DataFrame(columns=['cle','denomination','siren'])
cles_copro = set()

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
        m['cle']   = (
            m[c_sec].astype(str).str.strip() + '_' +
            m[c_num].astype(str).str.strip().str.zfill(4)
        )
        nb_prop    = m.groupby('cle').size().reset_index(name='nb_prop')
        cles_copro = set(nb_prop[nb_prop['nb_prop'] > 1]['cle'])
        cles_pp    = set(nb_prop[nb_prop['nb_prop'] == 1]['cle'])
        cols_keep  = ['cle'] + [c for c in [c_den, c_sir] if c]
        majic_pp   = m[m['cle'].isin(cles_pp)][cols_keep].copy()
        rename_map = {}
        if c_den:
            rename_map[c_den] = 'denomination'
        if c_sir:
            rename_map[c_sir] = 'siren'
        majic_pp = majic_pp.rename(columns=rename_map)
        if 'denomination' not in majic_pp.columns:
            majic_pp['denomination'] = ''
        if 'siren' not in majic_pp.columns:
            majic_pp['siren'] = ''
        MAJIC_OK = True
        print(f'✅ MAJIC — {len(cles_pp)} PP | {len(cles_copro)} copropriétés')
    else:
        print('⚠️ Colonnes section/numero introuvables')

    del m
    gc.collect()
except Exception as e:
    print(f'⚠️ MAJIC non chargé : {e}')


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


# Cartofriches CEREMA
print('Cartofriches...')
url_carto = 'https://static.data.gouv.fr/resources/sites-references-dans-cartofriches/20250422-125540/friches-standard.csv'
r = requests.get(url_carto, timeout=120)
r.raise_for_status()
carto_all = pd.read_csv(io.StringIO(r.text), sep=';', low_memory=False)

col_insee  = next((c for c in carto_all.columns if 'insee' in c.lower()), None)
col_nom    = next((c for c in carto_all.columns if 'nom' in c.lower() and 'site' in c.lower()), None)
col_adr    = next((c for c in carto_all.columns if 'adresse' in c.lower()), None)
col_surf   = next((c for c in carto_all.columns if 'surface' in c.lower()), None)
col_type   = next((c for c in carto_all.columns if 'type' in c.lower() and 'site' in c.lower()), None)
col_statut = next((c for c in carto_all.columns if 'statut' in c.lower()), None)
col_url    = next((c for c in carto_all.columns if 'url' in c.lower()), None)
col_prop   = next((c for c in carto_all.columns if 'proprio' in c.lower() or 'proprietaire' in c.lower()), None)

friches = carto_all[carto_all[col_insee].astype(str) == CODE_INSEE].copy()
print(f'✅ {len(friches)} friches à {CODE_INSEE}')

# Géolocalisation
col_pt = next((c for c in friches.columns if friches[c].astype(str).str.contains('POINT', na=False).any()), None)
col_lat = next((c for c in friches.columns if c.lower() in ('lat','latitude')), None)
col_lon = next((c for c in friches.columns if c.lower() in ('lon','lng','longitude')), None)

if col_pt:
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

if len(gdf_friches) > 0:
    print(f'✅ {len(gdf_friches)} friches géolocalisées')

# ══════════════════════════════════════════════════════
# DVF 2014–2023
# ══════════════════════════════════════════════════════
print(f'DVF {ANNEE_DVF_DEBUT}–{ANNEE_DVF_FIN}...')
dvf_frames = []

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
                        'Donation','Echange','Vente en l\'état futur d\'achèvement'
                    ])
                ].copy()
            dvf_frames.append(df)
            print(f'  {annee} ✅ {len(df)} mutations (maisons/terrains)')
        else:
            print(f'  {annee} ❌ {r.status_code}')
    except Exception as e:
        print(f'  {annee} ❌ {str(e)[:50]}')
    time.sleep(0.3)

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

    print(f'\n✅ {len(cles_avec_mutation)} parcelles avec mutation depuis {ANNEE_DVF_DEBUT}')
    print(f'   Types : {dvf["nature_mutation"].value_counts().to_dict()}')
else:
    cles_avec_mutation = set()
    dvf_last = pd.DataFrame()
    print('⚠️ DVF non chargé')


# Analyse : 4 catégories
print('Analyse...')

# Pleine propriété
if MAJIC_OK:
    parc_pp = parcelles.merge(
        majic_pp[['cle','denomination','siren']], on='cle', how='left'
    )
    parc_pp['denomination'] = parc_pp['denomination'].fillna('Particulier')
    parc_pp['siren']        = parc_pp['siren'].fillna('')
    parc_pp = parc_pp[~parc_pp['cle'].isin(cles_copro)].copy()
    masque = parc_pp['denomination'].str.upper().apply(
        lambda x: any(d in x for d in DOM_PUBLIC)
    )
    parc_pp = parc_pp[~masque].copy()
else:
    parc_pp = parcelles.copy()
    parc_pp['denomination'] = 'Particulier'
    parc_pp['siren']        = ''

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
              'emprise_ratio','compacite','denomination','siren']],
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

# ── Catégorie 4 : Biens vacants ─────────────────
candidats = parcelles.copy()
if MAJIC_OK:
    candidats = candidats[~candidats['cle'].isin(cles_copro)].copy()
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
candidats = candidats[~masque_pub].copy()
parc_vacants = candidats.merge(parc_geom, on='cle', how='left').copy()
parc_vacants['categorie'] = 'Bien vacant'
print(f'  🟤 {len(parc_vacants)} biens vacants')

print(f'\n✅ Total : {len(dc_parc)+len(vides)+len(sous)+len(gdf_friches)+len(parc_vacants)} opportunités')


def get_adresse(lat, lon):
    try:
        r = requests.get(
            f'https://api-adresse.data.gouv.fr/reverse/?lon={lon}&lat={lat}',
            timeout=5)
        feats = r.json().get('features', [])
        return feats[0]['properties'].get('label', '') if feats else ''
    except Exception:
        return ''


def geocoder_df(df):
    if len(df) == 0:
        df = df.copy()
        df['adresse'] = []
        return df
    pts = df.copy()
    if pts.crs and pts.crs.to_epsg() != 4326:
        pts = pts.to_crs(epsg=4326)
    pts['geometry'] = pts.geometry.centroid
    adrs = []
    for _, row in pts.iterrows():
        adrs.append(get_adresse(round(row.geometry.y, 6), round(row.geometry.x, 6)))
        time.sleep(0.1)
    df = df.copy()
    df['adresse'] = adrs
    return df


print('Géocodage...')
dc_parc      = geocoder_df(dc_parc)
vides        = geocoder_df(vides)
sous         = geocoder_df(sous)
parc_vacants = geocoder_df(parc_vacants)
if len(gdf_friches) > 0:
    gdf_friches = geocoder_df(gdf_friches)
print('✅ Adresses récupérées')


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


centre = parcelles.to_crs(epsg=4326).geometry.centroid.unary_union.centroid
carte  = folium.Map(location=[centre.y, centre.x], zoom_start=15)

folium.TileLayer(
    tiles=('https://wmts.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile'
           '&VERSION=1.0.0&LAYER=CADASTRALPARCELS.PARCELLAIRE_EXPRESS'
           '&STYLE=normal&FORMAT=image/png&TILEMATRIXSET=PM'
           '&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}'),
    attr='© IGN', name='Parcelles IGN', overlay=True, control=True, opacity=0.55
).add_to(carte)

fg_dc  = folium.FeatureGroup(name=f'🟠 Dents creuses ({len(dc_parc)})', show=True)
fg_vid = folium.FeatureGroup(name=f'🔵 Terrains vides stricts ({len(vides)})', show=True)
fg_sou = folium.FeatureGroup(name=f'🟣 Sous-exploités filtrés ({len(sous)})', show=True)
fg_fri = folium.FeatureGroup(name=f'🔴 Friches ({len(gdf_friches)})', show=True)
fg_vac = folium.FeatureGroup(name=f'🟤 Biens vacants ({len(parc_vacants)})', show=True)


def ajouter(fg, gp, lat, lon, coul, col_f, icone, html, tip, sec, num):
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
        gp, lat, lon = get_gp_latlon(row)
        gm, ge, sir_h = popup_base(rang,'DC',COUL_DC,adr,sec,num,surf,emp,prop,sir)
        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_DC}'>#{rang} Dent creuse | {niveau(haut)}</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Parcelle :</b> {sec} n°{num} | <b>Surface :</b> {surf} m²<br>"
            f"<b>Emprise :</b> {emp}% | <b>Haut :</b> {haut}m | <b>Voisin :</b> {vois}m | <b style='color:{COUL_DC}'>Écart : {ecar}m</b>"
            f"<hr style='margin:5px 0'><b>Proprio :</b> {prop} ({type_prop(prop,sir)})<br>{sir_h}"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        ajouter(fg_dc,gp,lat,lon,COUL_DC,'orange','arrow-up',html,f'#{rang} Dent creuse | {adr}',sec,num)
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
        gp, lat, lon = get_gp_latlon(row)
        gm, ge, sir_h = popup_base(rang,'Vide',COUL_VIDE,adr,sec,num,surf,emp,prop,sir)
        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_VIDE}'>#{rang} Terrain vide strict</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Parcelle :</b> {sec} n°{num} | <b>Surface :</b> {surf} m²<br>"
            f"<b>Emprise PCI :</b> {emp}% | <b>Bâti détecté :</b> {emp_m2} m² max"
            f"<hr style='margin:5px 0'><b>Proprio :</b> {prop} ({type_prop(prop,sir)})<br>{sir_h}"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        ajouter(fg_vid,gp,lat,lon,COUL_VIDE,'blue','tint',html,f'#{rang} Vide | {adr}',sec,num)
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
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        ajouter(fg_sou,gp,lat,lon,COUL_SOUS,'purple','building',html,f'#{rang} Sous-exp. | {adr}',sec,num)
    except Exception as e:
        print(f'  ⚠️ Sous {rang}: {e}')

# 🔴 Friches
print(f'Friches ({len(gdf_friches)})...')
for rang, (_, row) in enumerate(gdf_friches.iterrows(), 1):
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
        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_FRICHE}'>#{rang} Friche répertoriée</b><br>"
            f"<b>{nom}</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Type :</b> {typ} | <b>Statut :</b> {stat}<br>"
            f"<b>Surface :</b> {surf} m² | <b>Proprio :</b> {prop_f}<br>"
            f"<b>Source :</b> Cartofriches CEREMA"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a>"
            f"{'<br><br><a href=' + repr(url_f) + ' target=_blank style=font-size:11px;color:#666>Fiche →</a>' if url_f else ''}"
            f"</div>"
        )
        folium.Marker(
            [lat, lon], popup=folium.Popup(html, max_width=320),
            tooltip=f'#{rang} Friche | {nom}',
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(fg_fri)
        if geom.geom_type != 'Point':
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda x: {'color': COUL_FRICHE, 'weight': 2, 'fillOpacity': 0.3}
            ).add_to(fg_fri)
    except Exception as e:
        print(f'  ⚠️ Friche {rang}: {e}')

# 🟤 Biens vacants
print(f'Biens vacants ({len(parc_vacants)})...')
for rang, (_, row) in enumerate(parc_vacants.iterrows(), 1):
    try:
        sec = str(row.get('section','')).strip()
        num = str(row.get('numero','')).strip()
        surf = int(row['contenance']) if pd.notna(row.get('contenance')) else '?'
        adr = str(row.get('adresse','') or 'Adresse inconnue')
        prop = str(row.get('denomination','Particulier'))
        sir = str(row.get('siren','') or '')
        emp = round(float(row.get('emprise_ratio',0) or 0) * 100, 1)
        gp, lat, lon = get_gp_latlon(row)
        gm, ge, sir_h = popup_base(rang,'Vacant',COUL_VACANT,adr,sec,num,surf,emp,prop,sir)
        html = (
            f"<div style='font-family:Arial;font-size:13px;min-width:270px;line-height:1.9'>"
            f"<b style='font-size:15px;color:{COUL_VACANT}'>#{rang} Bien potentiellement vacant</b><br>"
            f"<a href='{gm}' target='_blank' style='color:#1a6fb5;font-weight:bold;text-decoration:none'>📍 {adr}</a><br><br>"
            f"<b>Parcelle :</b> {sec} n°{num} | <b>Surface :</b> {surf} m²<br>"
            f"<b>Emprise bâtie :</b> {emp}%"
            f"<hr style='margin:5px 0'><b>Proprio :</b> {prop} ({type_prop(prop,sir)})<br>{sir_h}"
            f"<hr style='margin:5px 0'>✓ Pleine propriété<br>✓ Aucune vente DVF depuis {ANNEE_DVF_DEBUT}<br>✓ Emprise modérée ({emp}%)"
            f"<hr style='margin:5px 0'><i style='font-size:11px;color:#888'>⚠️ À vérifier sur place</i>"
            f"<hr style='margin:5px 0'><a href='{ge}' target='_blank' style='background:#1a73e8;color:white;padding:5px 12px;border-radius:6px;text-decoration:none;font-size:12px'>🌍 Google Earth</a></div>"
        )
        ajouter(fg_vac,gp,lat,lon,COUL_VACANT,'darkred','question',html,f'#{rang} Vacant | {adr}',sec,num)
    except Exception as e:
        print(f'  ⚠️ Vacant {rang}: {e}')

for fg in [fg_dc, fg_vid, fg_sou, fg_fri, fg_vac]:
    fg.add_to(carte)
folium.LayerControl(collapsed=False, position='topright').add_to(carte)

total = len(dc_parc)+len(vides)+len(sous)+len(gdf_friches)+len(parc_vacants)
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
    f"<span style='color:{COUL_VACANT}'>&#9679;</span> Biens vacants {len(parc_vacants)}"
    f"</div>"
))

print(f'✅ Carte affichée — {total} opportunités')
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
        (dc_parc, 'Dent creuse'),
        (vides, 'Terrain vide'),
        (sous, 'Sous-exploité'),
        (gdf_friches, 'Friche'),
        (parc_vacants, 'Bien vacant'),
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
