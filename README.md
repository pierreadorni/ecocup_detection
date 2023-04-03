# Détection d'écocups

## Détection d'objets quelconques

Afin de détecter les objets possibles dans l'image, nous utilisons une technique de recherche sélective basée sur la méthode de segmentation de *Felzenszwalb* et *Huttenlocher*.

Cette technique est implémentée dans le module python `opencv-contrib-python`:

```python
import cv2
sss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
```

### Tests 

à tester: on pourrait récupérer la liste des objets détectés par SSS, et voir si un d'entre eux correspond à une BB d'ecocup (en utilisant IoU)

On pourrait ensuite comparer la précision de cette technique de détection d'objets avec la fenêtre glissante.
## Classification des objets

à faire