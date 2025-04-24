import json
import os.path

import torch
from einops import rearrange
import xml.etree.ElementTree as ET
from datetime import datetime
from vivit import ViViT
from dataset import preprocess_video
from utils.train_utils import *
import decord
import numpy as np
import warnings
from xml.dom import minidom

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def merge_labels(predictions):
    """Merge consecutive frames with the same label."""
    merged_labels = []
    if not predictions:
        return merged_labels

    current_sequence = {
        'frame_indexes': [predictions[0]['frame_index']],
        'start_frame_timestamp': predictions[0]['start_frame_timestamp'],
        'end_frame_timestamp': predictions[0]['end_frame_timestamp'],
        'label': predictions[0]['label']
    }

    for i in range(1, len(predictions)):
        item = predictions[i]
        if item['label'] == current_sequence['label']:
            # Extend the current sequence
            current_sequence['frame_indexes'].append(item['frame_index'])
            current_sequence['end_frame_timestamp'] = item['end_frame_timestamp']
        else:
            # Append the completed sequence and start a new one
            merged_labels.append(current_sequence)
            current_sequence = {
                'frame_indexes': [item['frame_index']],
                'start_frame_timestamp': item['start_frame_timestamp'],
                'end_frame_timestamp': item['end_frame_timestamp'],
                'label': item['label']
            }

    # Append the final sequence
    merged_labels.append(current_sequence)
    return merged_labels


def generate_eaf(merged_labels, output_file, video_path=""):
    """Generate an EAF file from the merged list of labels."""

    # Predefined CVE_IDs for labels
    # Used in original EAF files
    label_to_cveid = {
        "studio": "cveid_38d8d2cb-f9fe-4fb6-a0ba-ed1e535ab8ff",
        "indoor": "cveid_81e4eaaf-94ca-4a39-a6f7-22048483205f",
        "outdoor": "cveid_3409df0f-c4e1-4c63-ac3d-01d9334961bf",
        "předěl": "cveid_d0e20634-a77a-42bc-bbf1-d99f14c75dbf",
        "reklama": "cveid_c1efb72e-3a34-4baa-ba7d-597399c540af",
        "upoutávka": "cveid_2834383e-80c7-437d-80ef-c59f7e2ac59a",
        "grafika": "cveid_d9e8fbf1-d8b0-4d02-bbbc-767f6ce83308",
        "zábava": "cveid_e9842ad1-1199-404d-b8fa-0410df4ed937",
        "nedefinováno": "cveid_84147389-e1c0-4354-bd59-b9df4fdfad01"
    }

    # Initialize the XML structure
    annotation_doc = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "",
        "DATE": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "FORMAT": "3.0",
        "VERSION": "3.0",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/EAFv3.0.xsd"
    })

    # Add header
    header = ET.SubElement(annotation_doc, "HEADER", {
        "MEDIA_FILE": "",
        "TIME_UNITS": "milliseconds"
    })
    ET.SubElement(header, "MEDIA_DESCRIPTOR", {
        "MEDIA_URL": f"file://{video_path}",
        "MIME_TYPE": "video/mp4",
        "RELATIVE_MEDIA_URL": f"{video_path.split('/')[-1]}",
    })
    ET.SubElement(header, "PROPERTY", {"NAME": "URN"}).text = "urn:nl-mpi-tools-elan-eaf:generated"
    ET.SubElement(header, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(len(merged_labels))

    # Add time order
    time_order = ET.SubElement(annotation_doc, "TIME_ORDER")
    time_slot_map = {}
    for i, item in enumerate(merged_labels):
        ts_start_id = f"ts{i * 2 + 1}"
        ts_end_id = f"ts{i * 2 + 2}"
        time_slot_map[item['frame_indexes'][0]] = ts_start_id
        time_slot_map[item['frame_indexes'][-1]] = ts_end_id
        ET.SubElement(time_order, "TIME_SLOT", {
            "TIME_SLOT_ID": ts_start_id,
            "TIME_VALUE": str(int(item['start_frame_timestamp'] * 1000))
        })
        ET.SubElement(time_order, "TIME_SLOT", {
            "TIME_SLOT_ID": ts_end_id,
            "TIME_VALUE": str(int(item['end_frame_timestamp'] * 1000))
        })

    # Add tier and annotations
    tier = ET.SubElement(annotation_doc, "TIER", {
        "LINGUISTIC_TYPE_REF": "segmentV",
        "TIER_ID": "segmentV"
    })
    for idx, item in enumerate(merged_labels):
        annotation = ET.SubElement(tier, "ANNOTATION")

        if item['label'] not in label_to_cveid.keys():
            warnings.warn(f"Label '{item['label']}' not found. Using unknown_cveid instead.")

        alignable_annotation = ET.SubElement(annotation, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{idx + 1}",
            "CVE_REF": label_to_cveid.get(item['label'], "unknown_cveid"),
            "TIME_SLOT_REF1": time_slot_map[item['frame_indexes'][0]],
            "TIME_SLOT_REF2": time_slot_map[item['frame_indexes'][-1]]
        })
        ET.SubElement(alignable_annotation, "ANNOTATION_VALUE").text = item['label']

    # Add linguistic type
    ET.SubElement(annotation_doc, "LINGUISTIC_TYPE", {
        "CONTROLLED_VOCABULARY_REF": "RAVDAI",
        "GRAPHIC_REFERENCES": "false",
        "LINGUISTIC_TYPE_ID": "segmentV",
        "TIME_ALIGNABLE": "true"
    })

    # Add language
    ET.SubElement(annotation_doc, "LANGUAGE", {
        "LANG_DEF": "http://cdb.iso.org/lg/CDB-00130975-001",
        "LANG_ID": "und",
        "LANG_LABEL": "undetermined (und)"
    })

    # Add controlled vocabulary
    controlled_vocab = ET.SubElement(annotation_doc, "CONTROLLED_VOCABULARY", {
        "CV_ID": "RAVDAI"
    })
    ET.SubElement(controlled_vocab, "DESCRIPTION", {
        "LANG_REF": "und"
    })
    for label, cveid in label_to_cveid.items():
        cv_entry = ET.SubElement(controlled_vocab, "CV_ENTRY_ML", {
            "CVE_ID": cveid
        })
        ET.SubElement(cv_entry, "CVE_VALUE", {"LANG_REF": "und"}).text = label

    with open(output_file, "w", encoding="utf-8") as file:
        rough_string = ET.tostring(annotation_doc, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        file.write(reparsed.toprettyxml(indent="    "))
    print('Result saved to', output_file)


def predict_and_save_video(video_path: str, output_path: str):
    result = []
    # Load the video
    video_handler = decord.VideoReader(video_path, num_threads=1)
    # Iterate over frame with a context window
    for i in range(data_config['context_size'], len(video_handler) - data_config['context_size'] + 1, 25):
        indexes = list(range(i - data_config['context_size'], i + data_config['context_size'] + 1))
        video = list(video_handler.get_batch(indexes).asnumpy())
        processed_video = preprocess_video(video)
        processed_video = rearrange(np.stack(processed_video), 't h w c -> t c h w')
        processed_video = torch.from_numpy(processed_video).float().to(device)
        processed_video = processed_video.unsqueeze(0)

        prediction = model(processed_video,
                           padding_mask=torch.tensor([False] * processed_video.shape[1]).unsqueeze(0).cuda())
        predicted_class = prediction.argmax(dim=1)
        r = {'frame_index': i, 'start_frame_timestamp': video_handler.get_frame_timestamp(i)[0],
             'end_frame_timestamp': video_handler.get_frame_timestamp(i)[1], 'label': CLASSES[predicted_class]}
        result.append(r)
        if args.verbose:
            print(r)

    merged_labels = merge_labels(result)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    generate_eaf(merged_labels, os.path.join(output_path, video_basename + '.eaf'), video_path=video_path)


if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    assert args.demo_video, "Demo video must be specified"
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    eval_config = config['evaluation']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes

    print('Loading model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViViT(model_config).to(device)
    # Move non-trainable mask to the device
    model.temporal_transformer.cls_mask = model.temporal_transformer.cls_mask.to(device)

    # Load the model
    checkpoint = torch.load(eval_config['checkpoint'], weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded.')

    print('Processing data...')
    os.makedirs(args.demo_output_path, exist_ok=True)
    if args.demo_video_path:
        assert os.path.exists(args.demo_video_path), print('Test metadata file does not exist')
        print('Processing batch of demo videos from: {}'.format(args.demo_video_path))
        test_metadata = json.load(open(args.demo_video_path))
        for video in test_metadata:
            video_path = video['video']
            predict_and_save_video(video_path, args.demo_output_path)
    else:
        assert 'demo_video' in args, print('Demo video or demo video path must be specified')
        print('Processing single video: {}'.format(args.demo_video))
        predict_and_save_video(args.demo_video, args.demo_output_path)
