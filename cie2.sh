wget -c -q http://www.comune.torino.it/anagrafe/img/cie.png

mv "cie.png" cie2.png

python cie-to-json.py ./cie2.png --config_name "CIE-2.0"
