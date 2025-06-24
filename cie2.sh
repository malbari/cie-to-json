wget -c -q https://img.corrierecomunicazioni.it/wp-content/uploads/2020/10/Italian_electronic_ID_card.jpg

mv "Italian_electronic_ID_card.jpg" cie2.jpg

python cie-to-json.py ./cie2.jpg
