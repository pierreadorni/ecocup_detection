# Utilisation du script ecocup_detection.py

Ce script permet de generer la base d'images positive et négative pour le projet de détection d'Ecocups, apprendre un classifieur sur cette base d'apprentissage et utiliser ce classifieur sur de nouvelles données.  

Le script peut être utilisé en exécutant la commande suivante depuis un terminal :  
`python ecocup_detection.py [commande] [paramètres]`

## Commande "train"

Cette commande permet d'apprendre un classifieur au choix parmi SVM rbf, SVM poly, random forest, gboostn et adaboost.  

Elle nécessiste les paramètres suivants :
- `<svm/poly/rf/gboost/adaboost>`: le choix du modèle de classification à entraîner.
- `<positive_data_dir>`: le chemin du répertoire contenant les images positives.
- `<negative_data_dir>`: le chemin du répertoire contenant les images négatives.
- `<model_file>`: le chemin du fichier dans lequel le modèle entraîné doit être sauvegardé.

Exemple :  
`python ecocup_detection.py train svm ./data/positive ./data/negative ./models/model_svm.joblib`

## Commande "test"

Cette commande permet de tester un classifieur un utilisant des données de test.  

Elle nécessite les paramètres suivants :  
- `<classifier.joblib>`: le chemin du modèle entraîné.
- `<test_data_dir>`: le chemin du répertoire contenant les images de test.
- `<output_dir>`: le chemin du répertoire dans lequel les prédictions doivent être sauvegardées.
- `<results.csv>`: le chemin du fichier CSV dans lequel les résultats de la prédiction doivent être enregistrés.

## Commande "generate"

Cette commande permet de générer des images pour entraîner un modèle.  

Elle nécessite les paramètres suivants :  
- `<pos/neg>`: le choix entre la génération d'images positives ou négatives.
- `<num_images>`: le nombre d'images à générer (pour les images positives, cette valeur correspond au nombre d'image différente à générer pour chaque image de la base d'image positive).
-    `<save_dir>`: le chemin du répertoire dans lequel les images générées doivent être sauvegardées.
-    `<source_dir>` (pour la génération d'images négatives uniquement) : le chemin du répertoire contenant les images à utiliser pour la génération d'images négatives.

Exemples :  
- `python ecocup_detection.py generate pos 3 ./data/positive_augmented`
- `python ecocup_detection.py generate neg 500 ./data/negative_generated ./data/source_images`



