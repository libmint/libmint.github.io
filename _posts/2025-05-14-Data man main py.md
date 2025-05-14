```
import argparse
import os
import json
from datetime import datetime

from model_utils import request_model
from aw_data_converter.data_utils import convert_existing_data_with_prompts, fill_unknown_answers
from gui_tasks.element_description_task import ElementDescriptionTask
from gui_tasks.dense_captioning_task import DenseCaptioningTask
from gui_tasks.qa_task import QATask
from gui_tasks.set_of_mark_task import SetOfMarkTask
from gui_tasks.state_transition_captioning_task import StateTransitionCaptioningTask

def run_selected_tasks(args):
    image_paths = []
    if args.image_dir:
        image_paths = [os.path.join(args.image_dir, f)
                       for f in os.listdir(args.image_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    else:
        image_paths = [args.image]

    results = []
    is_output_dir = os.path.isdir(args.output) or args.output.endswith("/")

    for img in image_paths:
        print(f"[+] Processing {img}")
        elements = []
        marked_image = img

        if args.use_localizer:
            from localizer import localize_screen_with_ocr
            base_name = os.path.splitext(os.path.basename(img))[0]
            local_output_json = f"localizer_{base_name}.json"
            localization = localize_screen_with_ocr(img, "som_output", local_output_json)
            elements = localization["elements"]
            marked_image = localization["som_screenshot_path"]

        result = {
            "screenshot": marked_image,
            "metadata": {},
            "perception_tasks": {},
            "created_at": datetime.utcnow().isoformat()
        }

        if args.all or args.task == "element":
            if args.use_localizer and elements:
                result["perception_tasks"]["element_description"] = elements
            else:
                raw = ElementDescriptionTask(marked_image).run(request_model)
                result["perception_tasks"]["element_description"] = json.loads(raw)

        if args.all or args.task == "caption":
            caption = DenseCaptioningTask(marked_image, elements).run(request_model)
            result["perception_tasks"]["dense_captioning"] = caption

        if args.all or args.task == "qa":
            qa = QATask(marked_image, args.question, elements).run(request_model)
            result["perception_tasks"]["qa"] = {"question": args.question, "answer": qa}

        if args.all or args.task == "mark":
            marks = SetOfMarkTask(marked_image, elements).run(request_model)
            result["perception_tasks"]["set_of_mark"] = marks

        if (args.all or args.task == "state") and args.before and args.after:
            state_caption = StateTransitionCaptioningTask(args.before, args.after).run(request_model)
            result["perception_tasks"]["state_transition_captioning"] = {
                "before_image": args.before,
                "after_image": args.after,
                "caption": state_caption
            }

        if is_output_dir:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img))[0]
            task_prefix = args.task if not args.all else "all"
            out_path = os.path.join(args.output, f"{task_prefix}_{base_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[✓] Saved: {out_path}")
        else:
            results.append(result)

    if not is_output_dir:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
        print(f"[✓] Results saved: {args.output}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--image")
    run.add_argument("--image_dir")
    run.add_argument("--before")
    run.add_argument("--after")
    run.add_argument("--question", default="What is the main purpose of this screen?")
    run.add_argument("--output", required=True)
    run.add_argument("--use_localizer", action="store_true", help="Use OCR-based element localizer")
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=["element", "caption", "qa", "mark", "state"])
    group.add_argument("--all", action="store_true")

    convert = subparsers.add_parser("convert")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)

    fill = subparsers.add_parser("fill")
    fill.add_argument("--input", required=True)
    fill.add_argument("--image_dir", required=True)
    fill.add_argument("--output", required=True)

    localize = subparsers.add_parser("localize")
    localize.add_argument("--image_dir", required=True, help="Directory of screenshots to localize")
    localize.add_argument("--output_dir", default="som_output", help="Directory for marked images")
    localize.add_argument("--json_dir", default="som_json", help="Directory for element JSONs")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "run":
        run_selected_tasks(args)
    elif args.mode == "convert":
        convert_existing_data_with_prompts(args.input, args.output)
    elif args.mode == "fill":
        fill_unknown_answers(args.input, args.image_dir, args.output)
    elif args.mode == "localize":
        from localizer import localize_screen_with_ocr
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.json_dir, exist_ok=True)
        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img_name in image_files:
            img_path = os.path.join(args.image_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            json_path = os.path.join(args.json_dir, f"localized_{base_name}.json")
            print(f"[+] Localizing {img_path}")
            localize_screen_with_ocr(img_path, args.output_dir, json_path)
        print(f"[✓] Localized {len(image_files)} images.")
```
