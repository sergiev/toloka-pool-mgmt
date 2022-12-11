from typing import List

import toloka.client as toloka
import utils
from toloka.client.assignment import Assignment
from tqdm import tqdm

FSCORE_THD = 0.5

# class for handling submissions in the detection pool
class DetectionSubmittedHandler:
    def __init__(
        self,
        client: toloka.TolokaClient,
        verification_pool_id: str,
    ):
        self.client = client
        self.verification_pool_id = verification_pool_id

    # reject assignment and restrict worker from any further assignments
    def rejection(self, assignment: Assignment, input_image_path: str):
        reason = "Failed control task"
        self.client.set_user_restriction(
            toloka.user_restriction.AllProjectsUserRestriction(
                user_id=assignment.user_id,
                private_comment=f"{reason}: {input_image_path}",
            )
        )
        self.client.reject_assignment(
            assignment_id=assignment.id, public_comment=reason
        )

    # create new tasks for the verification pool
    def __call__(self, assignments: List[Assignment]) -> None:
        verification_tasks = []
        handle_suite_counter = {"filtered": 0, "passed": 0}
        for assignment in tqdm(assignments):
            noncontrol_tasks = []
            for task, solution in zip(assignment.tasks, assignment.solutions):
                input_image_path = task.input_values["image"]
                if task.known_solutions is None:
                    task_content = {
                        "image": input_image_path,
                        "assignment_id": assignment.id,
                    }
                    if "result" in solution.output_values:
                        task_content["selection"] = solution.output_values["result"]

                    noncontrol_tasks.append(
                        toloka.Task(
                            pool_id=self.verification_pool_id,
                            input_values=task_content,
                        )
                    )
                    continue

                gt = task.known_solutions[0].output_values
                guess = solution.output_values

                # 'path' bool = 'no objects to outline' checkbox
                guess_path = guess.get("path", False)
                gt_path = gt.get("path", False)

                # if user choose "no objects to outline" but objects were present (or vice versa)
                if guess_path != gt_path:
                    self.rejection(assignment, input_image_path)
                    handle_suite_counter["filtered"] += 1
                    # empty noncontrol_tasks for safety, no single task from this assignment should pass
                    noncontrol_tasks = []
                    break

                # if in this control task the right answer "no objects to outline" was selected
                if gt_path:
                    continue  # so just skip this control task

                # check similarity between boxes from gt and guess
                if utils.fscore(gt["result"], guess["result"]) < FSCORE_THD:
                    self.rejection(assignment, input_image_path)
                    handle_suite_counter["filtered"] += 1
                    # empty noncontrol_tasks for safety, no single task from this assignment should pass
                    noncontrol_tasks = []
                    break
            handle_suite_counter["passed"] += 1
            verification_tasks.extend(noncontrol_tasks)
        print("Detection handle results, suites:", handle_suite_counter)
        if verification_tasks:
            self.client.create_tasks(
                verification_tasks, allow_defaults=True, open_pool=False
            )
