# EIR

Bu yöntem için hazırlanan görüntü veritabanına [bu linkten](https://www.lamda.nju.edu.cn/data_MIMLimage.ashx) erişilebilir.

Yöntemi test etmek için aşağıdaki adımlar izlenmelidir.

# 1 
Görüntü veritabanını ve test görüntülerini indiriniz

# 2 
preprocess_Input.py ile veritabanı ve test görüntüleri için ayrı ayrı birer npy uzantılı dosyalar oluşturunuz. Tabi dosya yollarınızı kullandığınız ortama göre düzenlemeniz gerekir.

# 3 
Derin ağı eih.py ile eğitiniz. Eğitilen ağ parametreleri models isimli bir klasöre otomatik olarak kayıt edilecektir. Eğitimleri hash_bits = 16,32 ve 64 için tekrarlayınız.

# 4 
extractHashCodes.py ile, hem veritabanı hemde test görüntülerinin öznitelik vehash kodlarnı elde edebilirsiniz. Değerler hashCodes isimli klasöre otomatik kaydedilecektir.

#5
Elde edilen txt formatındaki öznitelik ve hash kodlarını matlab ortamında kullanmak için uygun hale getiriniz, yani txt'den mat formatına dönüştürünüz.

#6 
Evaluate_eihRetrieval.m ile excell dosyalarında bulunan sorgu gruplarını kullanarak map skorlarını elde edebilirsiniz.

# Repodaki Dosyaların Görevleri :
