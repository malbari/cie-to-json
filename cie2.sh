wget -c -q https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/CIE_%28fronte%29.jpg/960px-CIE_%28fronte%29.jpg

mv "960px-CIE_(fronte).jpg" cie2.jpg

python cie-to-json.py ./cie2.jpg
