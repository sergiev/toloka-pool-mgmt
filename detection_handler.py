from typing import List

import toloka.client as toloka
import utils
from toloka.client.assignment import Assignment
from tqdm import tqdm
import datetime

FSCORE_THD = 0.8

# class for handling submissions in the detection pool
class DetectionSubmittedHandler:
    def __init__(
        self,
        client: toloka.TolokaClient,
        verification_pool_id: str,
        ignore_filename: List = ()
    ):
        self.client = client
        self.verification_pool_id = verification_pool_id
        self.ignore_filename = set(ignore_filename)

    # reject assignment and restrict worker from any further assignments
    def rejection(self, assignment: Assignment, input_image_path: str):
        reason = f"Failed control task in assignment #{assignment.id}"
        self.client.set_user_restriction(
            toloka.user_restriction.AllProjectsUserRestriction(
                user_id=assignment.user_id,
                private_comment=f"{reason}: {input_image_path}",
                will_expire=datetime.datetime.utcnow() + datetime.timedelta(days=1)
            )
        )
        self.client.reject_assignment(
            assignment_id=assignment.id, public_comment=reason
        )

    # create new tasks for the verification pool
    def __call__(self, assignments: List[Assignment]) -> None:
        verification_tasks = []
        handle_suite_counter = {"filtered": 0, "passed": len(assignments)}
        for assignment in tqdm(assignments):
            noncontrol_tasks = []
            for task, solution in zip(assignment.tasks, assignment.solutions):
                input_image_path = task.input_values["image"]
                if input_image_path.split('/')[-1] in self.ignore_filename:
                    continue
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
                if guess_path != gt_path or (
                    not gt_path
                    and utils.fscore(gt["result"], guess["result"]) < FSCORE_THD
                ):
                    self.rejection(assignment, input_image_path)
                    handle_suite_counter["filtered"] += 1
                    # empty noncontrol_tasks for safety, no single task from this assignment should pass
                    noncontrol_tasks = []
                    break

            verification_tasks.extend(noncontrol_tasks)

        handle_suite_counter["passed"] -= handle_suite_counter["filtered"]
        print("Detection handle results, suites:", handle_suite_counter)
        if verification_tasks:
            self.client.create_tasks(
                verification_tasks, allow_defaults=True, open_pool=True
            )
