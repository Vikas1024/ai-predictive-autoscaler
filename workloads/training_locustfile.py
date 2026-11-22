import os
from locust import HttpUser, task, between, LoadTestShape


class AutoscalerTrainingUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(5)
    def cpu_workload(self):
        self.client.get("/work", name="/work")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")


class TrainingWorkloadShape(LoadTestShape):
    """
    Controlled concurrent-user profiles.
    The runner supplies TRAINING_PROFILE and TRAINING_MAX_USERS.
    """

    profile = os.getenv("TRAINING_PROFILE", "constant")
    max_users = int(os.getenv("TRAINING_MAX_USERS", "8"))
    low_users = max(2, max_users // 4)

    profiles = {
        "constant": [
            (0, 1),
            (20, max_users),
            (220, max_users),
            (240, 1),
        ],
        "ramp_up": [
            (0, 1),
            (180, max_users),
            (220, max_users),
            (240, 1),
        ],
        "spike": [
            (0, low_users),
            (60, low_users),
            (70, max_users),
            (100, max_users),
            (110, low_users),
            (220, low_users),
            (240, 1),
        ],
        "burst": [
            (0, low_users),
            (20, low_users),
            (30, max_users),
            (50, max_users),
            (70, low_users),
            (90, low_users),
            (100, max_users),
            (120, max_users),
            (140, low_users),
            (160, low_users),
            (170, max_users),
            (190, max_users),
            (210, low_users),
            (240, 1),
        ],
        "ramp_down": [
            (0, max_users),
            (30, max_users),
            (210, low_users),
            (240, 1),
        ],
    }

    def tick(self):
        elapsed = self.get_run_time()
        stages = self.profiles.get(self.profile)

        if stages is None:
            raise ValueError(f"Unknown workload profile: {self.profile}")

        if elapsed >= stages[-1][0]:
            return None

        for index in range(len(stages) - 1):
            start_time, start_users = stages[index]
            end_time, end_users = stages[index + 1]

            if start_time <= elapsed < end_time:
                progress = (
                    (elapsed - start_time) /
                    (end_time - start_time)
                )

                users = round(
                    start_users +
                    (end_users - start_users) * progress
                )

                return max(1, users), 4

        return max(1, stages[-1][1]), 4
