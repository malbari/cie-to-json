wget -c -q https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/CIE_%28fronte%29.jpg/960px-CIE_%28fronte%29.jpg

mv "960px-CIE_(fronte).jpg" cie3.jpg

python cie-to-json.py ./cie3.jpg --config_name "CIE-3.0"
