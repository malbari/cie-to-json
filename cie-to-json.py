import cv2
import easyocr
import sys
import argparse
import numpy as np
import json
import math
import re # Importato per la pulizia delle chiavi JSON

# Variabile globale per il debug mode
DEBUG_MODE = False

# Default filename for configuration
DEFAULT_CONFIG_FILE = "cie-to-json-config.json"

def debug_print(*args, **kwargs):
    """Stampa messaggi solo se DEBUG_MODE è True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

def clean_json_key(text):
    """Pulisce una stringa per usarla come chiave JSON (minuscolo, underscore per spazi, rimuove speciali)."""
    if not text:
        return ""
    # Sostituisce gli spazi con underscore, converte in minuscolo e rimuove caratteri non alfanumerici
    cleaned_text = re.sub(r'\s+', '_', text).lower()
    cleaned_text = re.sub(r'[^a-z0-9_]', '', cleaned_text)
    return cleaned_text

def convert_bbox_to_proportional_xywh(bbox_raw, img_width_px, img_height_px):
    """
    Converte un bbox di EasyOCR (lista di 4 punti pixel) in [x_prop, y_prop, w_prop, h_prop] proporzionali (0-1).
    Ritorna anche le coordinate min/max proporzionali per comodità.
    """
    x_coords = [p[0] for p in bbox_raw]
    y_coords = [p[1] for p in bbox_raw]

    x0_px = min(x_coords)
    y0_px = min(y_coords)
    x1_px = max(x_coords)
    y1_px = max(y_coords)

    width_px = x1_px - x0_px
    height_px = y1_px - y0_px

    # Conversione in proporzioni
    x0_prop = x0_px / img_width_px
    y0_prop = y0_px / img_height_px
    width_prop = width_px / img_width_px
    height_prop = height_px / img_height_px # Correzione: da 'h_prop' a 'height_px'

    x1_prop = x1_px / img_width_px
    y1_prop = y1_px / img_height_px

    return x0_prop, y0_prop, width_prop, height_prop, x1_prop, y1_prop

def detect_text_and_proportional_coords(image_source, langs=['it','en'], gpu=False, min_confidence=0.3, reader_obj=None):
    """
    Rileva il testo e restituisce bounding box già in coordinate proporzionali (0-1).
    Ritorna anche le dimensioni originali dell'immagine.
    """
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_source}")
    elif isinstance(image_source, np.ndarray):
        img = image_source
    else:
        raise TypeError("image_source deve essere un percorso di file (str) o un oggetto immagine NumPy (np.ndarray).")

    img_height_px, img_width_px = img.shape[:2]

    if reader_obj is None:
        reader = easyocr.Reader(langs, gpu=gpu)
    else:
        reader = reader_obj

    results = reader.readtext(img)
    proportional_detections = []
    for bbox_raw, text, conf in results:
        if conf >= min_confidence:
            x0_prop, y0_prop, w_prop, h_prop, x1_prop, y1_prop = convert_bbox_to_proportional_xywh(bbox_raw, img_width_px, img_height_px)
            proportional_detections.append({
                'text': text,
                'x0_prop': x0_prop,
                'y0_prop': y0_prop,
                'x1_prop': x1_prop,
                'y1_prop': y1_prop,
                'w_prop': w_prop,
                'h_prop': h_prop,
                'conf': conf,
                'original_bbox_raw': bbox_raw
            })
    return proportional_detections, img_width_px, img_height_px


def print_raw_proportional_boxes(detections):
    debug_print("\n=== BOX GREZZI ESTRATTI (Proporzionali) ===")
    sorted_detections = sorted(detections, key=lambda d: (d['y0_prop'], d['x0_prop']))
    for i, d in enumerate(sorted_detections):
        debug_print(f"[{i}] 📍BBOX: x0={d['x0_prop']:.4f}, y0={d['y0_prop']:.4f}, w={d['w_prop']:.4f}, h={d['h_prop']:.4f} — → '{d['text']}' (conf={d['conf']:.2f})")


def merge_adjacent_proportional_boxes(detections, merge_config):
    """
    Fonde i bounding box adiacenti in "linee" o "campi" logici.
    Opera esclusivamente su coordinate proporzionali.
    Accetta il dizionario di configurazione del merge.
    """
    MV_FACTOR = merge_config['MV_FACTOR']
    MO_WORD_FACTOR = merge_config['MO_WORD_FACTOR']
    MO_FIELD_FACTOR = merge_config['MO_FIELD_FACTOR']
    MO_COLUMN_FACTOR = merge_config['MO_COLUMN_FACTOR']
    PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE = merge_config['PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE']

    boxes = []
    for i, d in enumerate(detections):
        boxes.append({
            'text': d['text'],
            'x0_prop': d['x0_prop'],
            'y0_prop': d['y0_prop'],
            'x1_prop': d['x1_prop'],
            'y1_prop': d['y1_prop'],
            'w_prop': d['w_prop'],
            'h_prop': d['h_prop'],
            'conf': d['conf'],
            'original_idx': i,
            'original_bbox_raw': d['original_bbox_raw']
        })
    boxes.sort(key=lambda b: (b['y0_prop'], b['x0_prop']))

    merged_lines = []
    current_line_boxes = []

    current_line_y_sum = 0
    current_line_height_sum = 0
    current_line_max_x = 0

    for box in boxes:
        text, x0_prop, y0_prop, x1_prop, y1_prop, conf, h_prop = \
            box['text'], box['x0_prop'], box['y0_prop'], box['x1_prop'], box['y1_prop'], \
            box['conf'], box['h_prop']

        avg_line_height_prop = current_line_height_sum / len(current_line_boxes) if current_line_boxes else h_prop
        dynamic_mv = avg_line_height_prop * MV_FACTOR

        box_center_y_prop = (y0_prop + y1_prop) / 2
        avg_line_y_center_prop = current_line_y_sum / len(current_line_boxes) if current_line_boxes else box_center_y_prop

        if not current_line_boxes or \
           abs(box_center_y_prop - avg_line_y_center_prop) > dynamic_mv or \
           (y0_prop < current_line_boxes[-1]['y1_prop'] - PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE and x0_prop < current_line_boxes[-1]['x0_prop']):
            if current_line_boxes:
                merged_lines.append(current_line_boxes)
            current_line_boxes = [box]
            current_line_y_sum = box_center_y_prop
            current_line_height_sum = h_prop
            current_line_max_x = x1_prop
        else:
            gap_prop = x0_prop - current_line_max_x

            dynamic_mo_word = avg_line_height_prop * MO_WORD_FACTOR
            dynamic_mo_field = avg_line_height_prop * MO_FIELD_FACTOR
            dynamic_mo_column = avg_line_height_prop * MO_COLUMN_FACTOR

            if gap_prop < -PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE:
                if current_line_boxes:
                    merged_lines.append(current_line_boxes)
                current_line_boxes = [box]
                current_line_y_sum = box_center_y_prop
                current_line_height_sum = h_prop
                current_line_max_x = x1_prop
            elif gap_prop <= dynamic_mo_word:
                current_line_boxes.append(box)
                current_line_y_sum += box_center_y_prop
                current_line_height_sum += h_prop
                current_line_max_x = max(current_line_max_x, x1_prop)
            elif gap_prop <= dynamic_mo_field:
                current_line_boxes.append(box)
                current_line_y_sum += box_center_y_prop
                current_line_height_sum += h_prop
                current_line_max_x = max(current_line_max_x, x1_prop)
            elif gap_prop > dynamic_mo_column:
                if current_line_boxes:
                    merged_lines.append(current_line_boxes)
                current_line_boxes = [box]
                current_line_y_sum = box_center_y_prop
                current_line_height_sum += h_prop
                current_line_max_x = x1_prop
            else:
                current_line_boxes.append(box)
                current_line_y_sum += box_center_y_prop
                current_line_height_sum += h_prop
                current_line_max_x = max(current_line_max_x, x1_prop)

    if current_line_boxes:
        merged_lines.append(current_line_boxes)

    final_processed_boxes = []
    for line in merged_lines:
        line.sort(key=lambda b: b['x0_prop'])

        texts = [b['text'] for b in line]
        min_x_prop = min(b['x0_prop'] for b in line)
        min_y_prop = min(b['y0_prop'] for b in line)
        max_x_prop = max(b['x1_prop'] for b in line)
        max_y_prop = max(b['y1_prop'] for b in line)
        original_indices = [b['original_idx'] for b in line]
        original_raw_bboxes = [b['original_bbox_raw'] for b in line]

        merged_text = " ".join(texts)
        merged_box_props = {
            'x0_prop': min_x_prop,
            'y0_prop': min_y_prop,
            'x1_prop': max_x_prop,
            'y1_prop': max_y_prop,
            'w_prop': max_x_prop - min_x_prop,
            'h_prop': max_y_prop - min_y_prop
        }

        final_processed_boxes.append({
            'text': merged_text,
            'box_props': merged_box_props,
            'original_indices': original_indices,
            'original_raw_bboxes': original_raw_bboxes
        })

    return final_processed_boxes

def calculate_iou(box1_props, box2_props):
    """
    Calcola l'Intersection over Union (IoU) tra due bounding box proporzionali.
    Box props devono avere 'x0_prop', 'y0_prop', 'x1_prop', 'y1_prop'.
    """
    x_overlap = max(0, min(box1_props['x1_prop'], box2_props['x1_prop']) - max(box1_props['x0_prop'], box2_props['x0_prop']))
    y_overlap = max(0, min(box1_props['y1_prop'], box2_props['y1_prop']) - max(box1_props['y0_prop'], box2_props['y0_prop']))

    intersection_area = x_overlap * y_overlap

    box1_area = (box1_props['x1_prop'] - box1_props['x0_prop']) * (box1_props['y1_prop'] - box1_props['y0_prop'])
    box2_area = (box2_props['x1_prop'] - box2_props['x0_prop']) * (box2_props['y1_prop'] - box2_props['y0_prop'])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def calculate_center_distance(box1_props, box2_props):
    """
    Calcola la distanza euclidea tra i centri di due bounding box proporzionali.
    """
    center1_x = (box1_props['x0_prop'] + box1_props['x1_prop']) / 2
    center1_y = (box1_props['y0_prop'] + box1_props['y1_prop']) / 2
    center2_x = (box2_props['x0_prop'] + box2_props['x1_prop']) / 2
    center2_y = (box2_props['y0_prop'] + box2_props['y1_prop']) / 2
    return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

def extract_fields_by_position(processed_boxes, target_regions, extract_config):
    """
    Estrae i campi dalle regioni target usando il matching posizionale.
    Accetta il dizionario di configurazione per MIN_IOU_THRESHOLD.
    """
    extracted_data = {}
    
    available_boxes = list(processed_boxes)
    
    MIN_IOU_THRESHOLD = extract_config.get('MIN_IOU_THRESHOLD', 0.01) # Default 0.01 se non specificato

    for target_region in target_regions:
        json_key = target_region['json_key']
        target_bbox = target_region['bbox_target_prop']
        
        best_match_box = None
        best_iou = 0.0
        min_distance = float('inf')

        debug_print(f"\n[Extraction] Searching for '{json_key}' in target {target_bbox}")

        for i, p_box in enumerate(available_boxes):
            if p_box is None: # Se il box è già stato usato e impostato a None
                continue

            current_iou = calculate_iou(target_bbox, p_box['box_props'])
            current_distance = calculate_center_distance(target_bbox, p_box['box_props'])
            
            debug_print(f"  [Candidate] '{p_box['text']}' IoU: {current_iou:.4f}, Dist: {current_distance:.4f} BBOX: {p_box['box_props']}")

            if current_iou > MIN_IOU_THRESHOLD:
                if current_iou > best_iou:
                    best_iou = current_iou
                    min_distance = current_distance
                    best_match_box = p_box
                elif current_iou == best_iou and current_distance < min_distance:
                    min_distance = current_distance
                    best_match_box = p_box

        if best_match_box:
            extracted_data[json_key] = best_match_box['text']
            debug_print(f"[Extraction] Matched '{json_key}' with '{best_match_box['text']}' (IoU: {best_iou:.4f}, Dist: {min_distance:.4f})")
            
            for idx, item in enumerate(available_boxes):
                if item is not None and item['text'] == best_match_box['text'] and \
                   item['box_props']['x0_prop'] == best_match_box['box_props']['x0_prop'] and \
                   item['box_props']['y0_prop'] == best_match_box['box_props']['y0_prop']:
                   available_boxes[idx] = None
                   break
        else:
            extracted_data[json_key] = None
            debug_print(f"[Extraction] No match found for '{json_key}'.")

    return extracted_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrae testo e fonde i box OCR da immagini, operando solo con coordinate proporzionali. Esegue estrazione posizionale dei campi.")
    parser.add_argument("image_path", nargs='?', help="Percorso dell'immagine da processare (obbligatorio in modalità normale, facoltativo per --update_config).")
    parser.add_argument("--gpu", action="store_true", help="Usa la GPU per EasyOCR (se disponibile).")
    parser.add_argument("--debug", action="store_true", help="Abilita la stampa di messaggi di debug dettagliati.")
    parser.add_argument("--out", help="Specifica un file di output per il JSON (modalità normale).")
    parser.add_argument("--analyze", help="Se specificato, crea un file JSON con tutti i box mergiati per l'analisi e si ferma.")
    parser.add_argument("--update_config", help="Se specificato, aggiorna il file di configurazione con i dati del JSON di analisi fornito.")


    # Parametri di merge con default e override da riga di comando
    parser.add_argument("--mvf", type=float, default=0.5, help="MV_FACTOR (default: 0.5)")
    parser.add_argument("--mowf", type=float, default=0.7, help="MO_WORD_FACTOR (default: 0.7)")
    parser.add_argument("--moff", type=float, default=0.75, help="MO_FIELD_FACTOR (default: 0.75)")
    parser.add_argument("--mocf", type=float, default=0.75, help="MO_COLUMN_FACTOR (default: 0.75)")
    parser.add_argument("--pvo_tol", type=float, default=0.01, help="PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE (default: 0.01)")
    parser.add_argument("--min_iou_thresh", type=float, default=0.01, help="MIN_IOU_THRESHOLD for extraction (default: 0.01)")

    # Argomenti per il caricamento della configurazione da file (solo per modalità normale)
    parser.add_argument("--config_file", default=DEFAULT_CONFIG_FILE, help=f"Percorso del file di configurazione JSON. Default: {DEFAULT_CONFIG_FILE} (solo per modalità normale).")
    parser.add_argument("--config_name", default="CIE-1.0", help="Nome della configurazione da usare dal file JSON (es. 'CIE-1.0'). Default: 'CIE-1.0' (solo per modalità normale).")


    args = parser.parse_args()

    DEBUG_MODE = args.debug

    # --- Modalità UPDATE_CONFIG ---
    if args.update_config:
        debug_print(f"\n=== Modalità UPDATE_CONFIG abilitata. Caricamento file: {args.update_config} ===")
        try:
            with open(args.update_config, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        except FileNotFoundError:
            print(f"Errore: File di analisi '{args.update_config}' non trovato.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Errore: Il file '{args.update_config}' non è un JSON valido.")
            sys.exit(1)

        # Determina il nome della configurazione dal campo 'name' nel file di analisi, altrimenti dal nome del file
        config_name_to_update = analysis_data.get('name', args.update_config.split('/')[-1].replace('.json', ''))
        debug_print(f"Nome configurazione da aggiornare: '{config_name_to_update}'")

        image_width_px = analysis_data.get('image_width_px')
        image_height_px = analysis_data.get('image_height_px')
        merge_params_from_analysis = analysis_data.get('merge_params', {})
        merged_boxes_from_analysis = analysis_data.get('merged_boxes', [])

        if image_width_px is None or image_height_px is None:
            print("Errore: Il file di analisi non contiene 'image_width_px' o 'image_height_px'.")
            sys.exit(1)

        new_target_field_regions = []
        new_target_detect_regions = []

        # Popola TARGET_FIELD_REGIONS e TARGET_DETECT_REGIONS
        for box in merged_boxes_from_analysis:
            box_props = box['box_props']
            
            # Ensure x1_prop and y1_prop are present (they should be from analyze output)
            if 'x1_prop' not in box_props:
                box_props['x1_prop'] = round(box_props['x0_prop'] + box_props['w_prop'], 3)
            if 'y1_prop' not in box_props:
                box_props['y1_prop'] = round(box_props['y0_prop'] + box_props['h_prop'], 3)

            # TARGET_FIELD_REGIONS (se 'value' è valorizzato E 'detect' è 'f')
            if box.get('value') and box['value'].strip() != "" and box.get('detect', '').lower() == 'f':
                # Usa .copy() per assicurarsi che i box_props siano un nuovo oggetto nel JSON
                new_target_field_regions.append({
                    "json_key": clean_json_key(box['value']),
                    "bbox_target_prop": box_props.copy() # Usa una copia
                })
                debug_print(f"  Aggiunto a TARGET_FIELD_REGIONS: '{box.get('value')}'")

            # TARGET_DETECT_REGIONS (se 'detect' è 't')
            if box.get('detect', '').lower() == 't':
                # Determina la json_key: usa 'value' se presente e valorizzato, altrimenti 'text'
                key_for_detect_region = box.get('value')
                if not key_for_detect_region or key_for_detect_region.strip() == "":
                    key_for_detect_region = box['text']

                # Usa .copy() per assicurarsi che i box_props siano un nuovo oggetto nel JSON
                detect_region_entry = {
                    "json_key": clean_json_key(key_for_detect_region),
                    "bbox_target_prop": box_props.copy(), # Usa una copia
                    "text_original": box['text'] # Aggiunto il testo originale qui
                }
                new_target_detect_regions.append(detect_region_entry)
                debug_print(f"  Aggiunto a TARGET_DETECT_REGIONS: '{key_for_detect_region}' (Testo originale: '{box['text']}')")


        # Costruisci la nuova configurazione
        new_config_entry = {
            "MV_FACTOR": merge_params_from_analysis.get('MV_FACTOR', args.mvf),
            "MO_WORD_FACTOR": merge_params_from_analysis.get('MO_WORD_FACTOR', args.mowf),
            "MO_FIELD_FACTOR": merge_params_from_analysis.get('MO_FIELD_FACTOR', args.moff),
            "MO_COLUMN_FACTOR": merge_params_from_analysis.get('MO_COLUMN_FACTOR', args.mocf),
            "PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE": merge_params_from_analysis.get('PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE', args.pvo_tol),
            "MIN_IOU_THRESHOLD": args.min_iou_thresh, # Usa il default CLI o il valore sovrascritto
            "TARGET_FIELD_REGIONS": new_target_field_regions,
            "TARGET_DETECT_REGIONS": new_target_detect_regions # Nuova sezione
        }

        # Carica il file di configurazione esistente
        all_configs = {}
        try:
            with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                all_configs = json.load(f)
        except FileNotFoundError:
            debug_print(f"File di configurazione '{DEFAULT_CONFIG_FILE}' non trovato. Ne verrà creato uno nuovo.")
        except json.JSONDecodeError:
            print(f"Avviso: Il file '{DEFAULT_CONFIG_FILE}' non è un JSON valido o è vuoto. Verrà sovrascritto.")
            all_configs = {} # Inizializza come vuoto se non valido

        # Aggiungi o aggiorna la nuova configurazione
        all_configs[config_name_to_update] = new_config_entry

        # Salva il file di configurazione aggiornato
        try:
            with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_configs, f, indent=4, ensure_ascii=False)
            print(f"Configurazione '{config_name_to_update}' aggiornata con successo in '{DEFAULT_CONFIG_FILE}'.")
        except Exception as e:
            print(f"Errore durante la scrittura del file di configurazione: {e}", file=sys.stderr)
        
        sys.exit(0) # Termina lo script

    # --- Validazione dell'immagine solo per le modalità normali (non --update_config) ---
    if args.image_path is None:
        parser.error("L'argomento 'image_path' è obbligatorio in modalità normale (senza --analyze o --update_config).")

    # Carica l'immagine originale una volta
    original_image_full_res = cv2.imread(args.image_path)
    if original_image_full_res is None:
        print(f"Errore: Impossibile caricare l'immagine da {args.image_path}")
        sys.exit(1)

    # Inizializza EasyOCR Reader una volta
    ocr_reader = easyocr.Reader(['it', 'en'], gpu=args.gpu)

    # 1. Rilevamento dei box grezzi e conversione immediata in proporzioni
    detections_proportional, img_width_px, img_height_px = detect_text_and_proportional_coords(
        original_image_full_res, reader_obj=ocr_reader, min_confidence=0.3
    )
    debug_print(f"Dimensioni immagine originale: W={img_width_px}px, H={img_height_px}px")
    print_raw_proportional_boxes(detections_proportional)

    # Prepara la configurazione per il merge (da CLI o default)
    merge_config = {
        'MV_FACTOR': args.mvf,
        'MO_WORD_FACTOR': args.mowf,
        'MO_FIELD_FACTOR': args.moff,
        'MO_COLUMN_FACTOR': args.mocf,
        'PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE': args.pvo_tol
    }
    
    # 2. Fusione dei box adiacenti in "linee" o "campi" logici (lavora su proporzioni)
    processed_boxes = merge_adjacent_proportional_boxes(detections_proportional, merge_config)

    # --- Modalità ANALYZE ---
    if args.analyze:
        debug_print("\n=== Modalità ANALYZE abilitata. Creazione file JSON di analisi. ===")
        analysis_output_data = {
            "name": "analysis_output", # Default name for analysis output, can be changed by user for --update_config
            "image_width_px": img_width_px,
            "image_height_px": img_height_px,
            "merge_params": merge_config, # Aggiunto qui i parametri di merge
            "merged_boxes": []
        }
        
        # Ordina per una visualizzazione più leggibile
        sorted_processed_boxes_for_output = sorted(processed_boxes, key=lambda d: (d['box_props']['y0_prop'], d['box_props']['x0_prop']))
        
        for i, item in enumerate(sorted_processed_boxes_for_output):
            box_data = {
                "text": item['text'],
                "box_props": item['box_props'],
                "original_indices": item['original_indices'],
                "detect": "f", # Cambiato a minuscolo
                "value": ""
            }
            analysis_output_data["merged_boxes"].append(box_data)

        try:
            with open(args.analyze, 'w', encoding='utf-8') as f:
                json.dump(analysis_output_data, f, indent=4, ensure_ascii=False)
            print(f"File di analisi JSON scritto su: {args.analyze}")
        except Exception as e:
            print(f"Errore durante la scrittura del file di analisi: {e}", file=sys.stderr)
        sys.exit(0) # Termina lo script dopo aver creato il file di analisi
    
    # --- Modalità NORMALE (prosegue con l'estrazione) ---

    debug_print("\n=== BOX MERGED (Proporzionali) - Modalità Normale ===")
    # (La stampa di debug dei box mergiati è stata rimossa qui, dato che le informazioni complete sono nel file di analisi se `--analyze` è usato)
    
    # Carica la configurazione da file (solo se non in modalità analyze o update_config)
    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            all_configs = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File di configurazione '{args.config_file}' non trovato.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Errore: Il file '{args.config_file}' non è un JSON valido.")
        sys.exit(1)

    if args.config_name not in all_configs:
        print(f"Errore: La configurazione '{args.config_name}' non è presente nel file '{args.config_file}'.")
        print(f"Configurazioni disponibili: {', '.join(all_configs.keys())}")
        sys.exit(1)

    current_config = all_configs[args.config_name]
    debug_print(f"Caricata configurazione '{args.config_name}'.")

    config_target_regions = current_config['TARGET_FIELD_REGIONS']

    # Pre-calcola x1_prop e y1_prop per le regioni target
    for region in config_target_regions:
        target_bbox = region['bbox_target_prop']
        target_bbox['x1_prop'] = round(target_bbox['x0_prop'] + target_bbox['w_prop'], 3)
        target_bbox['y1_prop'] = round(target_bbox['y0_prop'] + target_bbox['h_prop'], 3)
    
    # Prepara la configurazione per l'estrazione (da CLI o default del JSON se presente)
    extract_config = {
        'MIN_IOU_THRESHOLD': args.min_iou_thresh # Usa il valore dalla CLI (default o override)
    }

    # 3. Estrazione dei campi tramite matching posizionale
    extracted_data = extract_fields_by_position(processed_boxes, config_target_regions, extract_config)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        print(f"Output JSON scritto su: {args.out}")
    else:
        json.dump(extracted_data, sys.stdout, indent=4, ensure_ascii=False)
        if not DEBUG_MODE:
            sys.stdout.write('\n')
