```
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run GUI perception tasks and output JSON.")

    parser.add_argument("--image", required=True, help="Main screenshot image path")
    parser.add_argument("--before", help="Before image path for state transition")
    parser.add_argument("--after", help="After image path for state transition")
    parser.add_argument("--question", default="What does the top button do?", help="Question for QA task")
    parser.add_argument("--output", default="gui_result.json", help="Output JSON filename")

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", choices=["element", "caption", "qa", "mark", "state"], help="Run only a specific task")
    task_group.add_argument("--all", action="store_true", help="Run all tasks")

    return parser.parse_args()

def run_selected_tasks(args):
    output = {
        "screenshot": args.image,
        "metadata": {},
        "perception_tasks": {},
        "created_at": datetime.utcnow().isoformat()
    }

    if args.all or args.task == "element":
        print("[+] Running element_description...")
        element_raw = ElementDescriptionTask(args.image).run()
        output["perception_tasks"]["element_description"] = parse_element_description(element_raw)

    if args.all or args.task == "caption":
        print("[+] Running dense_captioning...")
        caption = DenseCaptioningTask(args.image).run()
        output["perception_tasks"]["dense_captioning"] = caption

    if args.all or args.task == "qa":
        print("[+] Running QA...")
        qa = QATask(args.image, args.question).run()
        output["perception_tasks"]["qa"] = {
            "question": args.question,
            "answer": qa
        }

    if args.all or args.task == "mark":
        print("[+] Running set_of_mark_prompting...")
        marks = SetOfMarkTask(args.image).run()
        output["perception_tasks"]["set_of_mark_prompting"] = marks

    if (args.all or args.task == "state") and args.before and args.after:
        print("[+] Running state_transition_captioning...")
        state_caption = StateTransitionCaptioningTask(args.before, args.after).run()
        output["perception_tasks"]["state_transition_captioning"] = {
            "before_image": args.before,
            "after_image": args.after,
            "caption": state_caption
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[✓] 결과 저장 완료: {args.output}")

if __name__ == "__main__":
    args = parse_arguments()
    run_selected_tasks(args)

```
