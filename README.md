# Test GitHub de `Fusion_Final_v4`

Ce dossier te permet de tester ton notebook sur GitHub sans passer par Colab.

## Fichiers
- `run_fusion_github.py` : version script de ton notebook
- `.github/workflows/build-pages.yml` : lancement manuel depuis l'onglet Actions
- `requirements.txt` : dépendances Python
- `data/majic.parquet` : ton fichier MAJIC à déposer ici
- `docs/` : fichiers publiés par GitHub Pages

## Mise en place
1. Crée un repo GitHub.
2. Uploade tout le contenu du ZIP dans le repo.
3. Dépose ton fichier MAJIC dans `data/majic.parquet`.
4. Va dans **Settings > Pages** et choisis **GitHub Actions** comme source.
5. Va dans **Actions > Build and publish map > Run workflow**.
6. Laisse `93048` ou remplace par un autre code INSEE.
7. Une fois terminé, ouvre l'URL GitHub Pages.

## Remarques
- Si tu ne mets pas de MAJIC, le script tourne quand même, mais avec moins d'infos propriétaires.
- Le script génère `docs/index.html` et `docs/opportunites.geojson`.
- Si ton parquet MAJIC est trop gros pour GitHub normal, passe par Git LFS ou découpe-le.
