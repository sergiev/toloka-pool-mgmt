from collections import defaultdict
from typing import List

import pandas as pd  # To perform data manipulation
from crowdkit.aggregation import MajorityVote
from toloka.client.assignment import Assignment

# class for handling accepted tasks in the verification pool
class VerificationDoneHandler:
    def __init__(self, client, general_tasks_in_suite):
        self.microtasks = pd.DataFrame([], columns=["task", "label", "performer"])
        self.client = client
        self.blacklist = set()
        self.assignment_counter = defaultdict(
            lambda: {"accepted": 0, "rejected": 0, "total": general_tasks_in_suite}
        )
        self.path_to_assignment = dict()

    def update_blacklist(self, image_paths: set):
        self.blacklist.update(image_paths)

    # get the data necessary for aggregation
    def as_frame(self, assignments: List[Assignment]) -> pd.DataFrame:
        microtasks = []

        for assignment in assignments:
            for task, solution in zip(assignment.tasks, assignment.solutions):
                image_path = task.input_values["image"]
                if task.known_solutions is None and image_path not in self.blacklist:
                    microtasks.append(
                        (
                            image_path,  # task
                            solution.output_values["result"],  # label
                            assignment.user_id,  # worker
                        )
                    )
                    self.path_to_assignment[image_path] = task.input_values[
                        "assignment_id"
                    ]

        return pd.DataFrame(microtasks, columns=["task", "label", "worker"])

    # filter out tasks that already have enough overlap and aggregate the result
    def __call__(self, assignments: List[Assignment]) -> None:
        handle_task_counter = {"accepted": 0, "rejected": 0}
        # Initializing data
        microtasks = pd.concat([self.microtasks, self.as_frame(assignments)])

        # Filtering all microtasks that have overlap of 5
        microtasks["overlap"] = microtasks.groupby("task")["task"].transform("count")
        to_aggregate = microtasks[microtasks["overlap"] >= 5]
        microtasks = microtasks[microtasks["overlap"] < 5]
        aggregated = MajorityVote(
            default_skill=0.5, on_missing_skill="value"
        ).fit_predict(to_aggregate, None)

        # Accepting or rejecting assignments
        for image_path, result in aggregated.items():
            assignment_id = self.path_to_assignment[image_path]
            counter = self.assignment_counter[assignment_id]
            if result == "OK":
                counter["accepted"] += 1
                if counter["accepted"] == counter["total"]:
                    self.client.accept_assignment(assignment_id, "Well done!")
                    handle_task_counter["accepted"] += counter["total"]
            else:
                if counter["rejected"] == 0:  # prevent repeated rejection
                    self.client.reject_assignment(
                        assignment_id,
                        f"Some objects in {assignment_id} weren't selected or were selected incorrectly.",
                    )
                    handle_task_counter["rejected"] += counter["total"]
                counter["rejected"] += 1
        # Updating mictotasks
        self.microtasks = microtasks[["task", "label", "worker"]]
        print("Verification results, tasks:", handle_task_counter)
