import os.path
import xml.etree.ElementTree as ET
from datetime import datetime
from utils.train_utils import *
from xml.dom import minidom


def merge_two_eaf_files(reference_eaf, prediction_eaf, output_path):
    """Merge two EAF files into one with two separate tracks."""

    # Parse both files
    ref_tree = ET.parse(reference_eaf)
    pred_tree = ET.parse(prediction_eaf)
    ref_root = ref_tree.getroot()
    pred_root = pred_tree.getroot()

    # Extract time slots and tiers
    def extract_time_slots(root, prefix = ""):
        time_slots = {}
        for ts in root.find("TIME_ORDER").findall("TIME_SLOT"):
            ts_id = prefix + ts.attrib["TIME_SLOT_ID"]
            ts_value = int(ts.attrib["TIME_VALUE"])
            time_slots[ts_id] = ts_value
        return time_slots

    ref_time_slots = extract_time_slots(ref_root, prefix = "REF-")
    pred_time_slots = extract_time_slots(pred_root, prefix = "PRED-")

    def extract_tier_annotations(root, tier_id):
        annotations = []
        for tier in root.findall("TIER"):
            if tier.attrib["TIER_ID"] == tier_id:
                for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
                    annotations.append({
                        "id": annotation.attrib["ANNOTATION_ID"],
                        "start": annotation.attrib["TIME_SLOT_REF1"],
                        "end": annotation.attrib["TIME_SLOT_REF2"],
                        "value": annotation.find("ANNOTATION_VALUE").text
                    })
        return annotations

    ref_annotations = extract_tier_annotations(ref_root, "segmentV")
    pred_annotations = extract_tier_annotations(pred_root, "segmentV")

    # Merge time slots
    all_time_slots = {**ref_time_slots, **pred_time_slots}
    sorted_time_slots = sorted(all_time_slots.items(), key=lambda x: x[1])

    old2new_ts = {}
    new2old_ts = {}
    for i, items in enumerate(sorted_time_slots, start=1):
        ts, _ = items
        old2new_ts[ts] = "ts" + str(i)
        new2old_ts["ts" + str(i)] = ts

    # Assign new annotation IDs
    new_ref_annotations = []
    for i, ann in enumerate(ref_annotations, start=1):
        new_ref_annotations.append({
            "id": f"a{i}",
            "start": old2new_ts["REF-"+ann["start"]],
            "end": old2new_ts["REF-"+ann["end"]],
            "value": ann["value"]
        })

    start_id = len(new_ref_annotations) + 1
    new_pred_annotations = []
    for i, ann in enumerate(pred_annotations, start=start_id):
        new_pred_annotations.append({
            "id": f"a{i}",
            "start": old2new_ts["PRED-"+ann["start"]],
            "end": old2new_ts["PRED-"+ann["end"]],
            "value": ann["value"]
        })

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

    ET.SubElement(header, "MEDIA_DESCRIPTOR", dict(ref_root.find("HEADER").find("MEDIA_DESCRIPTOR").items()))
    ET.SubElement(header, "PROPERTY", {"NAME": "URN"}).text = "urn:nl-mpi-tools-elan-eaf:generated"
    ET.SubElement(header, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(len(new_ref_annotations) + len(new_pred_annotations))

    # Add time order
    time_order = ET.SubElement(header, "TIME_ORDER")
    for ts_id, ts_value in sorted_time_slots:
        ET.SubElement(time_order, "TIME_SLOT", {
            "TIME_SLOT_ID": old2new_ts[ts_id],
            "TIME_VALUE": str(ts_value)
        })

    # Add reference tier
    ref_tier = ET.SubElement(header, "TIER", {
        "LINGUISTIC_TYPE_REF": "segmentV",
        "TIER_ID": "reference"
    })
    for ann in new_ref_annotations:
        annotation = ET.SubElement(ref_tier, "ANNOTATION")
        alignable_annotation = ET.SubElement(annotation, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": ann["id"],
            "TIME_SLOT_REF1": ann["start"],
            "TIME_SLOT_REF2": ann["end"]
        })
        ET.SubElement(alignable_annotation, "ANNOTATION_VALUE").text = ann["value"]

    # Add prediction tier
    pred_tier = ET.SubElement(header, "TIER", {
        "LINGUISTIC_TYPE_REF": "segmentV",
        "TIER_ID": "prediction"
    })
    for ann in new_pred_annotations:
        annotation = ET.SubElement(pred_tier, "ANNOTATION")
        alignable_annotation = ET.SubElement(annotation, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": ann["id"],
            "TIME_SLOT_REF1": ann["start"],
            "TIME_SLOT_REF2": ann["end"]
        })
        ET.SubElement(alignable_annotation, "ANNOTATION_VALUE").text = ann["value"]

    # Add linguistic type and controlled vocabulary
    for elem in ref_root.findall(".//LINGUISTIC_TYPE"):
        header.append(elem)
    for elem in ref_root.findall(".//CONTROLLED_VOCABULARY"):
        header.append(elem)

    # Write to output file
    output_file = os.path.join(output_path, "merged.eaf")
    with open(output_file, "w", encoding="utf-8") as file:
        rough_string = ET.tostring(annotation_doc, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        prettyxml = reparsed.toprettyxml(indent="    ")
        prettyxml = prettyxml.rstrip()
        file.write(prettyxml)
    print('Result saved to', output_file)
    return output_file

if __name__ == "__main__":
    reference_eaf = "/media/zeleznyt/DATA/repo/ViViT/example_data_RAVDAI/ct24 2023-10-02 01.27.34_example.eaf"
    prediction_eaf = "/media/zeleznyt/DATA/repo/ViViT/example_data_RAVDAI/predictions.eaf"
    output_path = "/media/zeleznyt/DATA/repo/ViViT/example_data_RAVDAI/"
    output_file = merge_two_eaf_files(reference_eaf, prediction_eaf, output_path)