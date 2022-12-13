from collections import defaultdict
from typing import Sequence

import pandas as pd  # To perform data manipulation
from crowdkit.aggregation import MajorityVote
from toloka.client import TolokaClient
from toloka.client.assignment import Assignment
from tqdm import tqdm

# class for handling accepted tasks in the verification pool
class VerificationDoneHandler:
    def __init__(self, client: TolokaClient, general_tasks_in_suite: int):
        self.client = client
        self.blacklist = set()
        self.assignment_summary = defaultdict(
            lambda: {
                "accepted": set(),
                "rejected": False,
                "total": general_tasks_in_suite,
            }
        )
        self.path_to_assignment = dict()

    # get the data necessary for aggregation
    def as_frame(self, assignments: Sequence[Assignment]) -> pd.DataFrame:
        microtasks = []

        for assignment in assignments:
            for task, solution in zip(assignment.tasks, assignment.solutions):
                image_path = task.input_values["image"]
                detection_suite_id = task.input_values["assignment_id"]
                if (
                    task.known_solutions is None
                    and detection_suite_id not in self.blacklist
                ):
                    microtasks.append(
                        (
                            image_path,  # task
                            solution.output_values["result"],  # label
                            assignment.user_id,  # worker
                        )
                    )
                    self.path_to_assignment[image_path] = detection_suite_id

        return pd.DataFrame(microtasks, columns=["task", "label", "worker"])

    # filter out tasks that already have enough overlap and aggregate the result
    def __call__(self, assignments: Sequence[Assignment]) -> None:
        # Initializing data
        microtasks = self.as_frame(assignments)

        # Filtering all microtasks that have overlap of 5
        microtasks["overlap"] = microtasks.groupby("task")["task"].transform("count")
        to_aggregate = microtasks[microtasks["overlap"] >= 5]
        aggregated = MajorityVote(
            default_skill=0.5, on_missing_skill="value"
        ).fit_predict(to_aggregate, None)

        # Accepting or rejecting assignments
        handle_task_counter = {"accepted": 0, "rejected": 0}
        for image_path, result in tqdm(aggregated.items()):
            assignment_id = self.path_to_assignment[image_path]
            if assignment_id in self.blacklist:
                continue
            summary = self.assignment_summary[assignment_id]
            if result == "OK":
                summary["accepted"].add(image_path)
                if len(summary["accepted"]) == summary["total"]:
                    self.client.accept_assignment(assignment_id, "Well done!")
                    handle_task_counter["accepted"] += summary["total"]
                    self.blacklist.add(assignment_id)

            # prevent repeated rejection
            elif not summary["rejected"]:
                public_comment = (
                    "Majority of 5 verificators decided that some objects in "
                    + "this suite weren't selected or were selected incorrectly."
                )
                self.client.reject_assignment(
                    assignment_id,
                    public_comment,
                )
                handle_task_counter["rejected"] += summary["total"]
                summary["rejected"] = True
                self.blacklist.add(assignment_id)
        # Updating mictotasks
        print("Verification results, tasks:", handle_task_counter)
