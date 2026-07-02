# 🔎 Category Search — Google Product Type taxonomy (it-IT)

Moteur de recherche qui, à partir d'un produit, retourne les **catégories Google
les plus proches** (ID + chemin complet + scores). Interface en anglais.

- **Corpus** : taxonomie officielle Google `it-IT` (version 2021-09-21), 5 595 catégories.
- **Moteur hybride** :
  1. **Traduction** de la requête en italien (FR/EN → IT) via `deep-translator` — cross-lingue.
  2. **Lexical** : TF-IDF char n-grammes (gère pluriels/fautes, ex. `custodia`→`Custodie`).
  3. **Sémantique** : embeddings multilingues (`paraphrase-multilingual-MiniLM-L12-v2`) + cosinus.
  4. **Fusion RRF** pondérée par un curseur *Lexical ⟷ Semantic*.
- **UI** : Streamlit — filtre par rayon, nb de résultats, traduction on/off, exemples cliquables.

## Produit via l'URL

Le produit à classer peut être passé directement dans l'URL (lien partageable) :

```
http://localhost:8501/?q=running+shoes
https://<ton-app>.streamlit.app/?q=chitarra+elettrica
```

Le champ se remplit automatiquement et la recherche se lance. Toute recherche
faite dans l'UI met aussi à jour l'URL.

---

## Lancer en local

```bash
cd taxo-search
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python build_index.py          # (recommandé) précalcule embeddings.npy → démarrage instantané
streamlit run app.py
```

Ouvre http://localhost:8501.

> Sans `build_index.py`, l'app calcule les embeddings au **premier** lancement
> (~15–30 s) puis les met en cache dans `embeddings.npy`.

---

## Déployer gratuitement sur Streamlit Community Cloud

1. Pousse ce dossier sur un dépôt GitHub (avec `taxonomy.it-IT.txt` et, idéalement,
   `embeddings.npy` précalculé) :

   ```bash
   git init && git add . && git commit -m "Category search — Google taxonomy (it-IT)"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```

2. Va sur **https://share.streamlit.io** → connecte GitHub.
3. **New app** → dépôt, branche `main`, fichier `app.py` → **Deploy**.
4. URL publique `https://<...>.streamlit.app` (le premier build prend quelques minutes).

### Conseils free tier
- **Committe `embeddings.npy`** (~8 Mo) : au runtime le modèle ne sert plus qu'à encoder la requête.
- La traduction utilise l'endpoint gratuit Google Translate ; si le réseau échoue,
  l'app bascule silencieusement sur la requête brute (décoche « Translate » pour du 100 % local).

---

## Adapter à une autre langue / version

Remplace `taxonomy.it-IT.txt` par un autre fichier Google au même format
`ID - Chemin > ...` (ex. `taxonomy-with-ids.fr-FR.txt`), supprime `embeddings.npy`,
puis relance `python build_index.py`.

## Structure

```
taxo-search/
├── app.py               # application Streamlit (moteur hybride)
├── build_index.py       # précalcul des embeddings
├── taxonomy.it-IT.txt   # données Google
├── embeddings.npy       # généré (cache des vecteurs denses)
├── requirements.txt
└── README.md
```
