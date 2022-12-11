import time

import toloka.client as toloka

from detection_handler import DetectionSubmittedHandler
from sensitive import PRIVATE_TOKEN
from verification_handler import VerificationDoneHandler

PERIOD = 60  # seconds
GENERAL_TASKS_IN_DETECTION_SUITE = 2

if __name__ == "__main__":
    tc = toloka.TolokaClient(PRIVATE_TOKEN, "PRODUCTION")
    print(tc.get_requester())
    dpid = "36743071"
    vpid = "36799000"

    dsh = DetectionSubmittedHandler(client=tc, verification_pool_id=vpid)
    vdh = VerificationDoneHandler(
        client=tc, general_tasks_in_suite=GENERAL_TASKS_IN_DETECTION_SUITE
    )
    # DSH should ignore suits that already went to verification
    passed_detection_assignment_ids = set()
    # VDH should ignore images that already processed, i.e. accepted/rejected in detection stage
    verification_ignore_paths = set()

    while True:
        start = time.time()
        submitted_detection_assignments = list(
            tc.get_assignments(pool_id=dpid, status="SUBMITTED")
        )

        # handle submitted detections (which are not in verification pool yet)
        all_verification_tasks = tc.get_tasks(pool_id=vpid)
        for task in all_verification_tasks:
            passed_detection_assignment_ids.add(task.input_values["assignment_id"])
        unprocessed_detection_assignments = [
            i
            for i in submitted_detection_assignments
            if i.id not in passed_detection_assignment_ids
        ]
        print(
            f"""Detection pool handling starts: 
        {len(submitted_detection_assignments)} are pending verdict in the pool,
        {len(unprocessed_detection_assignments)} are to be processed now"""
        )
        dsh(unprocessed_detection_assignments)

        # handle accepted verifications (which are not already handled)
        processed_detection_assignments = list(
            tc.get_assignments(pool_id=dpid, status="REJECTED")
        ) + list(tc.get_assignments(pool_id=dpid, status="ACCEPTED"))

        for assignment in processed_detection_assignments:
            for task in assignment.tasks:
                verification_ignore_paths.add(task.input_values["image"])

        accepted_verification_assignments = list(
            tc.get_assignments(pool_id=vpid, status="ACCEPTED")
        )
        vdh.update_blacklist(verification_ignore_paths)
        print(
            f"Verification pool handling starts: {len(accepted_verification_assignments)} suites"
        )
        vdh(accepted_verification_assignments)

        sleep_duration = max(0, PERIOD - (time.time() - start))
        print(f"iteration finished, sleeping for {sleep_duration} seconds")
        time.sleep(sleep_duration)
