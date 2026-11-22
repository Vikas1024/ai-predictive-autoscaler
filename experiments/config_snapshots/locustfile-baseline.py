from locust import HttpUser, task, between


class AutoscalerUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(5)
    def cpu_workload(self):
        self.client.get("/work", name="/work")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
