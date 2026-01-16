import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, color, morphology, filters, measure, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import pandas as pd


# 1 Φόρτωση Εικόνας και Προεπεξεργασία
image=io.imread('normal40_var_refl.png')


# Μετατροπή σε Grayscale αν χρειάζεται
if len(image.shape) == 3:
    gray_image = color.rgb2gray(image)
else:
    gray_image = image


# Top-hat filtering (αντίστοιχο του imtophat στο Matlab)
# Αφαιρεί τον ανομοιόμορφο φωτισμό. Το 15 είναι η ακτίνα του δομικού στοιχείου.
#Αφαιρεί τις αργές μεταβολές φωτεινότητας του υπόβαθρου (background), εξομαλύνοντας 
# #τον φωτισμό και αναδεικνύοντας τα φωτεινά αντικείμενα (πυρήνες).
#Επειδή ο φωτισμός στο φόντο της εικόνας είναι ανομοιόμορφος, το Top-hat αφαιρεί αυτό το σκοτεινό υπόβαθρο, κάνοντας τους πυρήνες 
#να φαίνονται πιο φωτεινοί και ομοιόμορφοι σε όλη την εικόνα. Αυτό είναι απαραίτητο για να πετύχει το επόμενο βήμα.
selem = morphology.disk(15)
tophat_image = morphology.white_tophat(gray_image, selem)

# 2 Τμηματοποίηση (Segmentation) - Ερώτημα Α
# Δυαδικοποίηση (Binarization - imbinarize)
#Αυτή η μέθοδος βρίσκει αυτόματα την καλύτερη τιμή κατωφλίου (threshold) που διαχωρίζει τα pixels σε δύο ομάδες: Πυρήνας (φωτεινά pixels) και Φόντο (σκοτεινά pixels). 
# Το αποτέλεσμα είναι μια απλή, ασπρόμαυρη μάσκα (binary) που δείχνει την αρχική θέση των κυττάρων.
thresh= filters.threshold_otsu(tophat_image) #υπολογισμός του βέλτιστου κατωφλίου (threshold) για τον διαχωρισμό αντικειμένων από το φόντο.
binary= tophat_image > thresh #Δημιουργία δυαδικής μάσκας όπου τα pixels με τιμή μεγαλύτερη του κατωφλίου θεωρούνται κύτταρα.

# Αφαίρεση μικρών θορύβων (bwareaopen)
binary=morphology.remove_small_objects(binary, min_size=10)

# Υπολογισμός Distance Transform (bwdist)
#Κάθε pixel παίρνει τιμή ίση με την απόστασή του από το πλησιέστερο pixel του φόντου. Οι υψηλότερες τιμές (κορυφές) αντιστοιχούν στα κέντρα των πυρήνων.
distance=ndi.distance_transform_edt(binary)

# Watershed αλγόριθμος
# Βρίσκουμε τις κορυφές (peaks) της απόστασης για να φτιάξουμε markers
# *** ΔΙΟΡΘΩΣΗ 1: Προσθέτουμε min_distance=10 για καλύτερους markers ***

#: Χρησιμοποιούμε τη συνάρτηση peak_local_max για να εντοπίσουμε τις κορυφές στον Χάρτη Απόστασης.
coords=peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary, min_distance=10)#Σημαντικό: Η παράμετρος min_distance=10 ορίζει ότι δύο κορυφές πρέπει να απέχουν τουλάχιστον 10 pixels. Αυτό αποτρέπει την εύρεση 
# πολλαπλών κέντρων στον ίδιο πυρήνα
mask=np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)]=True
markers, _=ndi.label(mask)#Δημιουργία πίνακα markers όπου κάθε κορυφή παίρνει ένα μοναδικό ID.

# Εφαρμογή watershed
labels = segmentation.watershed(-distance, markers, mask=binary)

#Αφαίρεση μικρών αντικειμένων (θορύβου) μετά το Watershed ***
# Αυτό μειώνει τον υπερβολικό αριθμό αντικειμένων από 10228 σε ρεαλιστικό αριθμό.
cleaned_labels = morphology.remove_small_objects(labels, min_size=50)
#: Μετεπεξεργασία για αφαίρεση θορύβου. Αντικείμενα με εμβαδόν μικρότερο από 50 pixels διαγράφονται, καθώς δεν αποτελούν πραγματικούς πυρήνες.


# 3 Εξαγωγή Χαρακτηριστικών - Ερώτημα Β
# regionprops (ίδια με Matlab)
# *** Χρησιμοποιούμε τα καθαρισμένα labels (cleaned_labels) ***
props=measure.regionprops(cleaned_labels)
data=[]
print(f"Συνολικός αριθμός πυρήνων: {len(props)}")


for prop in props:
    # Επειδή το remove_small_objects μπορεί να αφαιρέσει τα labels, 
    # ελέγχουμε αν η περιοχή είναι > 0
    if prop.area == 0:
        continue
        
    # α) Εμβαδόν
    area=prop.area
    # β)Περίμετρος
    perimeter=prop.perimeter
    # γ) Πηλίκο ημιαξόνων (Major / Minor)
    # Προστασία για διαίρεση με το μηδέν
    if prop.minor_axis_length > 0:
        ratio = prop.major_axis_length / prop.minor_axis_length
    else:
        ratio = 0

    data.append([prop.label, area, perimeter, ratio])
    
# Αποθήκευση σε DataFrame
df=pd.DataFrame(data,columns=['Cell_ID', 'Area', 'Perimeter', 'Axis_Ratio'])
print(df)

# 4 Οπτικοποίηση (Visualization)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original Grayscale')

ax[1].imshow(tophat_image, cmap='gray')
ax[1].set_title('Top-hat Filtered')


# Χρωματισμός των labels (label2rgb)
# *** Χρησιμοποιούμε τα καθαρισμένα labels (cleaned_labels) ***
colored_labels = color.label2rgb(cleaned_labels, bg_label=0)
ax[2].imshow(colored_labels)
ax[2].set_title('Watershed Segmentation')


# Αρίθμηση πάνω στην εικόνα
ax[3].imshow(colored_labels)
ax[3].set_title('Αρίθμηση Πυρήνων')
for prop in props:
    y, x = prop.centroid
    ax[3].text(x, y, str(prop.label), color='white', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()