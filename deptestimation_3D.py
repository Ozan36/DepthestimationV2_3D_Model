# Gerekli kütüphanelerin eklenmesi
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QHBoxLayout, QWidget, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os
# Open3D kütüphanesini ekleme
import open3d as o3d   
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Derinlik tahmin modeli import etme
from depthv2model import DepthEstimator   


class DepthEstimationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Pencere başlığı
        self.setWindowTitle("Depth Estimation Viewer")   
        # Pencere boyutu ve konumu
        self.setGeometry(100, 100, 1000, 500)   
        # Derinlik tahminci nesnesi
        self.estimator = DepthEstimator()   

        self.initUI()  # Kullanıcı arayüzünü başlatma

    def initUI(self):
        # Butonları ve etiketleri oluşturma
        # Resim yükleme butonu
        self.loadButton = QPushButton("Resim Yükle", self)   
        # Butona tıklanma olayı
        self.loadButton.clicked.connect(self.loadImage)   
        # 3D model oluşturma butonu
        self.modelButton = QPushButton("3D Model Oluştur", self)  
        self.modelButton.clicked.connect(self.create3DModel)   

        # Etiketler
        # Orijinal resim etiketi
        self.originalLabel = QLabel("Orijinal Resim")   
        # Etiketi ortalama
        self.originalLabel.setAlignment(Qt.AlignCenter)   
        # Derinlik haritası etiketi
        self.depthLabel = QLabel("Derinlik Haritası")   
        self.depthLabel.setAlignment(Qt.AlignCenter) 

        # Layout oluşturma
        hbox = QHBoxLayout()  # Yatay yerleşim
        hbox.addWidget(self.originalLabel)  # Orijinal resmi yerleştir
        hbox.addWidget(self.depthLabel)  # Derinlik haritasını yerleştir
        
        vbox = QVBoxLayout()  # Dikey yerleşim
        vbox.addWidget(self.loadButton)  # Yükleme butonunu ekle
        vbox.addWidget(self.modelButton)  # Model oluşturma butonunu ekle
        vbox.addLayout(hbox)  # Yatay yerleşim ekle

        container = QWidget()  # Ana container
        container.setLayout(vbox)  # Dikey yerleşimi ayarla
        self.setCentralWidget(container)  # Merkez widget’ını ayarla

    def loadImage(self):
        # Resim yükleme fonksiyonu
        fileName, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resimler (*.png *.xpm *.jpg *.jpeg)")
        if fileName:
            # Orijinal resmi göster
            self.setImage(self.originalLabel, cv2.imread(fileName))

            # Derinlik tahmini yap
            self.depth_map = self.estimator.predict(fileName)
            self.depth_colored = cv2.applyColorMap(self.depth_map, cv2.COLORMAP_MAGMA)  # Renk haritası uygula
            self.setImage(self.depthLabel, self.depth_colored)  # Derinlik haritasını göster

    def create3DModel(self):
        # 3D model oluşturma fonksiyonu
        if hasattr(self, 'depth_map'):
            # Derinlik haritasını kullanarak 3D model oluşturma
            h, w = self.depth_map.shape  # Derinlik haritasının boyutlarını al
            x, y = np.meshgrid(np.arange(w), np.arange(h))  # Meshgrid oluştur
            z = self.depth_map / 255.0  # Derinlik haritasını normalize et

            # Nokta bulutu oluştur
            points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)  # Noktaları yığ
            points = points[~np.isnan(points).any(axis=1)]  # NaN değerleri kaldır

            # Open3D ile nokta bulutunu oluştur
            pcd = o3d.geometry.PointCloud()  # Nokta bulutu nesnesi
            pcd.points = o3d.utility.Vector3dVector(points)  # Noktaları atama

            # Nokta bulutunu görüntüle
            o3d.visualization.draw_geometries([pcd])  # Nokta bulutunu göster
        else:
            print("Önce bir resim yükleyin ve derinlik haritasını oluşturun.")  # Hata mesajı

    def setImage(self, label, img):
        # Resmi bir etikete set etme fonksiyonu
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
        h, w, ch = rgb_image.shape  # Resmin boyutlarını al
        bytes_per_line = ch * w  # Bir satırda byte sayısını hesapla
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # QImage oluştur
        pixmap = QPixmap.fromImage(qimg)  # QPixmap oluştur
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))  # Resmi etiket boyutuna göre ölçekle


if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication nesnesi oluştur
    mainWin = DepthEstimationApp()  # Uygulama penceresini başlat
    mainWin.show()  # Pencereyi göster
    sys.exit(app.exec_())  # Uygulamayı başlat ve çıkışta temizle
