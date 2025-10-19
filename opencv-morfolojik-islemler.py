import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- 1. Görüntüyü Yükleme ve Gri Tonlamaya Çevirme ---
image_path = "./resimler/coins.jpg"
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Hata: Resim yüklenemedi. Yol doğru mu? {image_path}")
    exit()

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# --- 2. İkili Görüntüye Çevirme (Binarization) ---
# Otsu'nun eşikleme yöntemi ile madeni paraları beyaz (ön plan), arka planı siyah yapıyoruz.
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# --- 3. Çekirdek (Kernel) Tanımlama ---
# 5x5 boyutunda bir çekirdek tanımlayalım. Boyutları deneysel olarak değiştirebiliriz.
kernel = np.ones((5, 5), np.uint8)

# --- 4. Erozyon (Erosion) ---
# Nesnelerin (paraların) kenarlarını aşındırır, küçük gürültüleri temizleyebilir.
erosion_image = cv2.erode(binary_image, kernel, iterations=1)

# --- 5. Genleşme (Dilation) ---
# Nesnelerin (paraların) kenarlarını genişletir, Erozyon'un tersidir.
dilation_image = cv2.dilate(binary_image, kernel, iterations=1)

# --- 6. Açma (Opening) ---
# Erozyon ardından Genleşme. Küçük beyaz gürültüleri temizlemek için idealdir.
opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2) # 2 iterasyon ile daha etkili temizlik

# --- 7. Kapama (Closing) ---
# Genleşme ardından Erozyon. Nesnelerin içindeki küçük delikleri doldurur.
closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2) # 2 iterasyon ile daha etkili doldurma

# --- 8. Morfolojik Gradyan (Morphological Gradient) ---
# Kenar algılama için kullanılır: (Genleşme - Erozyon)
gradient_image = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)

# --- 9. Top Hat (Beyaz Şapka) ---
# Orijinalden açma işleminin çıkarılması, parlak küçük detayları vurgular.
# Genellikle gri tonlamalı görüntü üzerinde yapılır.
tophat_image = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)

# --- 10. Coin Sınırlarını Bulma (Final Adım) ---
# Gradyan, zaten kenarları iyi vurgular. Bunu kullanarak kenarları çizelim.
# Gradyan görüntüsünü renkli orijinal görüntü üzerine uygulayacağız.

# Gradyan görüntüsü 0-255 arasındadır. Kenarlar beyazdır.
# Bu kenarları orijinal renkli resme kırmızı olarak çizmek istiyoruz.
result_image = original_image.copy() # Orijinal renkli resmin bir kopyasını al
# Gradyan görüntüsündeki beyaz pikseller (kenarlar) için orijinal resimde kırmızı renk ata
result_image[gradient_image == 255] = [0, 0, 255] # BGR formatında Kırmızı: [0, 0, 255]

# --- Tüm Sonuçları Matplotlib ile Görselleştirme ---
titles = [
    'Orijinal Gri', 'İkili Görüntü', 'Erozyon', 'Genleşme',
    'Açma (Opening)', 'Kapama (Closing)', 'Morfolojik Gradyan', 'Top Hat', 'Sonuç (Kenarlar Kırmızı)'
]
images = [
    gray_image, binary_image, erosion_image, dilation_image,
    opening_image, closing_image, gradient_image, tophat_image, result_image
]

plt.figure(figsize=(15, 10)) # Büyük bir figür boyutu ile daha net gösterim

for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    # Renkli görüntüler için 'cmap' belirtmeyiz, Matplotlib otomatik olarak işler.
    # Gri tonlamalı görüntüler için 'cmap='gray'' kullanırız.
    if len(images[i].shape) == 2 or titles[i] == 'Sonuç (Kenarlar Kırmızı)': # Gri veya kenarlı görüntü
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
    else: # Renkli görüntüler için BGR'yi RGB'ye çevir
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([]) # Eksen işaretlerini gizle

plt.tight_layout() # Alt grafikleri otomatik düzenle
plt.show()