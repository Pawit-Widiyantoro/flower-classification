# Tugas Pengenalan Pola - Klasifikasi Bunga
## Deskripsi Project

- Dataset yang digunakan merupakan dataset flower-recognition yang diambil dari platform kaggle. Dataset berisi hampir 4.000 gambar yang terbagi ke dalam 5 kategori yaitu daisy, dandelion, rose, sunflower, tulip. Data disimpan di dalam folder dataset di direktori root dari program utama, akan tetapi dataset dimasukkan ke dalam file .gitignore sehingga tidak di-push ke repository.
![WhatsApp Image 2024-06-11 at 21 56 12_3cbe0d16](https://github.com/Pawit-Widiyantoro/flower-classification/assets/146058025/b785cff9-0a1f-4ca1-879a-14bf18f6b61f)

- Data kemudian diload ke dalam program menggunakan file data.py yang disimpan di folder utils.
- Model disimpan di file utama yaitu main.py.
- Proses feature recognition/pengenalan ciri dilakukan secara otomati oleh arsitektur CNN yang dibangun pada tiap layernya.
- Layer konvolusi bertugas dalam proses feature extraction/pengambilan ciri dimana setiap fitur/ciri dari gambar diekstraksi ke dalam bentuk numerik. firtu yang sudah dalam bentuk numerik ini kemudian dapat digunakan untuk pemrosesan seperti klasifikasi. Fitur yang diekstraksi dapat berupa warna, teksture, dll. 
- Feature recognition/pengenalan ciri bertujuan untuk mengambil fitur yang memiliki pengaruh besar atau penting terhadap keseluruhan data. Proses ini dilakukan pada pooling layer dan fully connected layer.
- Model yang digunakan dibangun secara manual dengan menambahkan layer konvolusi dan dropout ke arsitektur CNN yang telah dibagikan sebelumnya. Training hanya dilakukan dalam 10 epoch mengingat banyaknya jumlah data yang diproses.
- Proses evaluasi menunjukkan dari 10 epoch yang dijalankan, model berhasil belajar dengan baik jika dilihat dari nilai loss tiap epoch nya yang semakin menurun. Nilai akurasi setiap epoch juga semakin meningkat. Hal ini menunjukkan jika epoch ditambah, maka terdapat kemungkinan nilai akurasi yang didapat menjadi lebih tinggi.
![image](https://github.com/Rosita-Dian-Febriyanti/klasifikasi_bunga/assets/146058025/754b57d4-a941-408d-8af6-ceba6d6b3fa4)

## Anggota Kelompok 
1. Pawit Widiyantoro        (21102184)
2. Rosita Dian Febriyanti   (21102186)

## Kontribusi Anggota
1. Pawit Widiyantoro        (21102184)
   - Membuat program 
3. Rosita Dian Febriyanti   (21102186)
   - Mencari dataset
