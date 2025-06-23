import cv2
import easyocr
import sys
import argparse
import numpy as np
import json
import math

# Variabile globale per il debug mode
DEBUG_MODE = False

def debug_print(*args, **kwargs):
    """Stampa messaggi solo se DEBUG_MODE è True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

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
    height_prop = height_px / img_height_px

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


def merge_adjacent_proportional_boxes(detections, config):
    """
    Fonde i bounding box adiacenti in "linee" o "campi" logici.
    Opera esclusivamente su coordinate proporzionali.
    Accetta il dizionario di configurazione.
    """
    MV_FACTOR = config['MV_FACTOR']
    MO_WORD_FACTOR = config['MO_WORD_FACTOR']
    MO_FIELD_FACTOR = config['MO_FIELD_FACTOR']
    MO_COLUMN_FACTOR = config['MO_COLUMN_FACTOR']
    PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE = config['PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE']

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
    # Determinare le coordinate dell'intersezione
    x_overlap = max(0, min(box1_props['x1_prop'], box2_props['x1_prop']) - max(box1_props['x0_prop'], box2_props['x0_prop']))
    y_overlap = max(0, min(box1_props['y1_prop'], box2_props['y1_prop']) - max(box1_props['y0_prop'], box2_props['y0_prop']))

    intersection_area = x_overlap * y_overlap

    # Calcolare l'area di ciascun box
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

def extract_fields_by_position(processed_boxes, target_regions, config):
    """
    Estrae i campi dalle regioni target usando il matching posizionale.
    Accetta il dizionario di configurazione per MIN_IOU_THRESHOLD.
    """
    extracted_data = {}
    
    # Copia dei box processati per poter rimuovere quelli già usati
    available_boxes = list(processed_boxes)
    
    MIN_IOU_THRESHOLD = config.get('MIN_IOU_THRESHOLD', 0.01) # Default 0.01 se non specificato

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
            
            # Rimuovi il box utilizzato dalla lista per evitare duplicati
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
    parser.add_argument("image_path", help="Percorso dell'immagine da processare.")
    parser.add_argument("--gpu", action="store_true", help="Usa la GPU per EasyOCR (se disponibile).")
    parser.add_argument("--debug", action="store_true", help="Abilita la stampa di messaggi di debug dettagliati.")
    parser.add_argument("--out", help="Specifica un file di output per il JSON. Se omesso, stampa su stdout.")
    parser.add_argument("--config_file", default="cie-to-json-config.json", help="Percorso del file di configurazione JSON. Default: cie-to-json-config.json")
    parser.add_argument("--config_name", default="CIE-1.0", help="Nome della configurazione da usare dal file JSON (es. 'CIE-1.0'). Default: 'CIE-1.0'")


    args = parser.parse_args()

    DEBUG_MODE = args.debug

    # Carica il file di configurazione
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

    # Estrai le configurazioni specifiche
    config_merge_params = {
        'MV_FACTOR': current_config['MV_FACTOR'],
        'MO_WORD_FACTOR': current_config['MO_WORD_FACTOR'],
        'MO_FIELD_FACTOR': current_config['MO_FIELD_FACTOR'],
        'MO_COLUMN_FACTOR': current_config['MO_COLUMN_FACTOR'],
        'PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE': current_config['PROPORTIONAL_VERTICAL_OVERLAP_TOLERANCE']
    }
    config_target_regions = current_config['TARGET_FIELD_REGIONS']

    # Pre-calcola x1_prop e y1_prop per le regioni target per comodità nei calcoli IoU
    for region in config_target_regions:
        target_bbox = region['bbox_target_prop']
        target_bbox['x1_prop'] = round(target_bbox['x0_prop'] + target_bbox['w_prop'], 3)
        target_bbox['y1_prop'] = round(target_bbox['y0_prop'] + target_bbox['h_prop'], 3)
    
    # Aggiungi MIN_IOU_THRESHOLD alla configurazione specifica per l'estrazione posizionale
    # Questo valore può essere messo nel JSON se si vuole configurarlo per tipo di CIE.
    config_extract_params = {
        'MIN_IOU_THRESHOLD': current_config.get('MIN_IOU_THRESHOLD', 0.01)
    }

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

    # 2. Fusione dei box adiacenti in "linee" o "campi" logici (lavora su proporzioni)
    processed_boxes = merge_adjacent_proportional_boxes(detections_proportional, config_merge_params)

    debug_print("\n=== BOX MERGED (Proporzionali) ===")
    sorted_processed_boxes = sorted(processed_boxes, key=lambda d: (d['box_props']['y0_prop'], d['box_props']['x0_prop']))
    for i, item in enumerate(sorted_processed_boxes):
        text = item['text']
        box_props = item['box_props']
        original_indices = item['original_indices']
        debug_print(f"[{i}] 📍BBOX (prop): x0={box_props['x0_prop']:.4f}, y0={box_props['y0_prop']:.4f}, w={box_props['w_prop']:.4f}, h={box_props['h_prop']:.4f} — → '{text}' (Merged from raw indices: {original_indices})")

    # 3. Estrazione dei campi tramite matching posizionale
    extracted_data = extract_fields_by_position(processed_boxes, config_target_regions, config_extract_params)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        print(f"Output JSON scritto su: {args.out}")
    else:
        json.dump(extracted_data, sys.stdout, indent=4, ensure_ascii=False)
        if not DEBUG_MODE:
            sys.stdout.write('\n')