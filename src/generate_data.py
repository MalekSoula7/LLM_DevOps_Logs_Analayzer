"""Synthetic CI/CD log generator for the DevOps Log Analyzer.

Generates 600 labeled training samples and 150 drift samples across 5 failure
categories, with realistic multi-line log snippets mimicking GitHub Actions,
GitLab CI, Jenkins, and Kubernetes output.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_DRIFT = ROOT / "data" / "drift"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_DRIFT.mkdir(parents=True, exist_ok=True)

RUNNER_IDS = ["runner-01", "runner-02", "runner-03", "gha-ubuntu-latest", "gl-runner-k8s"]
REPOS = ["myapp/backend", "myapp/frontend", "platform/api-gateway", "infra/terraform-modules", "svc/auth-service"]
BRANCHES = ["main", "develop", "feature/user-auth", "fix/payment-bug", "release/v2.1"]
PIPELINE_IDS = [f"#{random.randint(1000, 9999)}" for _ in range(200)]

NPM_PACKAGES = ["lodash", "react", "axios", "express", "webpack", "typescript", "jest", "@types/node", "eslint"]
PIP_PACKAGES = ["django", "fastapi", "sqlalchemy", "celery", "redis", "boto3", "numpy", "pandas", "requests"]
JAVA_PACKAGES = ["com.fasterxml.jackson", "org.springframework", "io.micronaut", "org.hibernate"]
TS_FILES = ["src/components/UserDashboard.tsx", "src/api/client.ts", "src/utils/validators.ts", "src/types/index.ts"]
PY_FILES = ["src/models/user.py", "tests/test_api.py", "src/utils/auth.py", "src/services/payment.py"]
ERROR_CODES = [1, 2, 127, 128, 137, 139, 143]


def _ts(delta_minutes: int = 0) -> str:
    """Generate a realistic ISO-like log timestamp offset from a base time."""
    base = datetime.now() - timedelta(days=random.randint(0, 30))
    t = base + timedelta(minutes=delta_minutes)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _pipeline_header(label: str) -> list[str]:
    """Return standard CI pipeline log header lines."""
    t = _ts(0)
    repo = random.choice(REPOS)
    branch = random.choice(BRANCHES)
    run_id = random.choice(PIPELINE_IDS)
    runner = random.choice(RUNNER_IDS)
    return [
        f"[{t}] Pipeline run {run_id} started",
        f"[{t}] Repository: {repo} | Branch: {branch}",
        f"[{t}] Runner: {runner}",
    ]


# ---------------------------------------------------------------------------
# BUILD FAILURE GENERATORS
# ---------------------------------------------------------------------------

_NPM_ERRORS = [
    lambda pkg: [
        "npm warn deprecated {pkg}@3.x: This version is no longer supported.".format(pkg=pkg),
        "npm error code ENOENT",
        "npm error syscall open",
        "npm error path /home/runner/work/{repo}/node_modules/.cache".format(repo=random.choice(REPOS)),
        "npm error errno -2",
        "npm error enoent ENOENT: no such file or directory, open 'package-lock.json'",
        "npm error A complete log of this run can be found in: /home/runner/.npm/_logs/",
    ],
    lambda pkg: [
        f"npm install {pkg}",
        f"npm ERR! code E404",
        f"npm ERR! 404 Not Found - GET https://registry.npmjs.org/{pkg} - Not found",
        f"npm ERR! 404 '{pkg}@latest' is not in the npm registry.",
        "npm ERR! You should bug the author to publish it (or use the name yourself!)",
        "npm ERR! A complete log of this run can be found in:",
    ],
    lambda pkg: [
        f"added 847 packages in 23s",
        f"",
        f"npm warn peer dependencies:",
        f"npm warn {pkg}@2.1.0 requires a peer of react@^17 but none is installed.",
        f"npm ERR! code ERESOLVE",
        f"npm ERR! ERESOLVE unable to resolve dependency tree",
        f"npm ERR! Found: react@18.2.0",
        f"npm ERR! Could not resolve dependency: peer react@'^17.0.0' from {pkg}@2.1.0",
    ],
]

_PIP_ERRORS = [
    lambda pkg: [
        f"Collecting {pkg}",
        f"  Downloading {pkg}-1.2.3.tar.gz (245 kB)",
        f"  Preparing metadata (setup.py) ... error",
        f"  error: subprocess-exited-with-error",
        f"  × python setup.py egg_info did not run successfully.",
        f"  │ exit code: 1",
        f"  ╰─> [15 lines of output]",
        f"      error: command '/usr/bin/gcc' failed with exit code 1",
        f"ERROR: Could not build wheels for {pkg}, which is required to install pyproject.toml-based projects",
    ],
    lambda pkg: [
        f"Collecting {pkg}==99.99.0",
        f"  ERROR: Could not find a version that satisfies the requirement {pkg}==99.99.0",
        f"  (from versions: 1.0.0, 1.1.0, 1.2.3, 2.0.0)",
        f"ERROR: No matching distribution found for {pkg}==99.99.0",
    ],
    lambda pkg: [
        f"Requirement already satisfied: pip in /usr/local/lib/python3.11/site-packages",
        f"Collecting {pkg}",
        f"  Downloading {pkg}-3.1.4-cp311-cp311-linux_x86_64.whl.metadata (4.1 kB)",
        f"  ERROR: RECORD file has an incorrect hash: expected sha256=...",
        f"  The RECORD file checksum does not match. Please uninstall and reinstall.",
        f"ERROR: pip's dependency resolver does not currently take into account all packages",
        f"ERROR: pip subprocess to install build dependencies exited with 1",
    ],
]

_MAVEN_ERRORS = [
    lambda pkg: [
        "[INFO] Scanning for projects...",
        "[INFO] ------------------------------------------------------------------------",
        "[INFO] BUILD FAILURE",
        "[INFO] ------------------------------------------------------------------------",
        f"[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.11.0:compile",
        f"[ERROR] on project {pkg.split('.')[-1]}: Compilation failure",
        "[ERROR] /src/main/java/com/app/Service.java:[42,8] error: cannot find symbol",
        "[ERROR]   symbol:   class UserRepository",
        "[ERROR] -> [Help 1]",
    ],
]

_TS_ERRORS = [
    lambda f: [
        f"$ tsc --noEmit",
        f"{f}(45,12): error TS2345: Argument of type 'string | undefined' is not assignable",
        f"  to parameter of type 'string'.",
        f"{f}(78,5): error TS2339: Property 'userId' does not exist on type 'Request'.",
        f"{f.replace('.tsx', '.ts') if '.tsx' in f else f}(103,18): error TS7016: Could not find a declaration file for module 'legacy-lib'.",
        f"Found 3 errors in 2 files.",
        f"error Command failed with exit code 1.",
    ],
]

_DOCKER_BUILD_ERRORS = [
    lambda _: [
        "Step 5/12 : RUN pip install -r requirements.txt",
        " ---> Running in a3f2b1c9d4e5",
        "Collecting uvicorn==0.30.6",
        "  ERROR: pip's subprocess to install build dependencies exited with 1",
        "The command '/bin/sh -c pip install -r requirements.txt' returned a non-zero code: 1",
        "ERROR: failed to build: exit code: 1",
    ],
]


def generate_build_failure_log() -> str:
    """Generate a realistic build failure log entry."""
    lines = _pipeline_header("build_failure")
    t = _ts(1)
    lines.append(f"[{t}] Stage: build")

    r = random.random()
    if r < 0.30:
        pkg = random.choice(NPM_PACKAGES)
        lines.append(f"[{t}] Running: npm install")
        tmpl = random.choice(_NPM_ERRORS)
        lines.extend(tmpl(pkg))
    elif r < 0.55:
        pkg = random.choice(PIP_PACKAGES)
        lines.append(f"[{t}] Running: pip install -r requirements.txt")
        tmpl = random.choice(_PIP_ERRORS)
        lines.extend(tmpl(pkg))
    elif r < 0.68:
        pkg = random.choice(JAVA_PACKAGES)
        lines.append(f"[{t}] Running: mvn package -DskipTests")
        tmpl = random.choice(_MAVEN_ERRORS)
        lines.extend(tmpl(pkg))
    elif r < 0.82:
        f = random.choice(TS_FILES)
        lines.append(f"[{t}] Running: npm run build")
        tmpl = random.choice(_TS_ERRORS)
        lines.extend(tmpl(f))
    else:
        lines.append(f"[{t}] Running: docker build -t myapp:latest .")
        tmpl = random.choice(_DOCKER_BUILD_ERRORS)
        lines.extend(tmpl(None))

    exit_code = random.choice(ERROR_CODES[:4])
    t2 = _ts(2)
    lines.append(f"[{t2}] ERROR: Stage 'build' failed with exit code {exit_code}")
    lines.append(f"[{t2}] Pipeline run aborted.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TEST FAILURE GENERATORS
# ---------------------------------------------------------------------------

TEST_FILES = [
    "tests/test_api.py", "tests/test_auth.py", "tests/test_database.py",
    "tests/test_payment.py", "tests/test_utils.py", "tests/integration/test_e2e.py",
]
TEST_NAMES = [
    "test_create_user", "test_login_valid_credentials", "test_payment_processing",
    "test_delete_record", "test_get_user_by_id", "test_invalid_token",
    "test_rate_limiting", "test_db_connection", "test_send_email",
]
HTTP_CODES = [400, 401, 403, 404, 500, 503]

_PYTEST_FAILURES = [
    lambda n, f: [
        f"collected {random.randint(30, 80)} items",
        f"",
        f"FAILED {f}::{n} - AssertionError: assert {random.choice(HTTP_CODES)} == 200",
        f"FAILED {f}::test_list_items - AssertionError: assert [] != []",
        f"ERROR {random.choice(TEST_FILES)}::test_setup - fixture 'db_session' not found",
        f"",
        f"short test summary info",
        f"FAILED {f}::{n}",
        f"FAILED {f}::test_list_items",
        f"{'=' * 50}",
        f"{random.randint(2, 8)} failed, {random.randint(10, 40)} passed in {random.uniform(2.0, 30.0):.2f}s",
    ],
    lambda n, f: [
        f"FAILED {f}::{n}",
        f"_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _",
        f"",
        f"    def {n}(client):",
        f">       response = client.post('/api/users', json={{\"email\": \"test@example.com\"}})",
        f"E       AssertionError: assert <Response [500 INTERNAL SERVER ERROR]> == 200",
        f"E       +  where <Response [500]> = client.post(...)",
        f"",
        f"{f}:{random.randint(40, 120)}: AssertionError",
        f"{'=' * 50}",
        f"FAILED {f}::{n} - AssertionError",
    ],
    lambda n, f: [
        f"Running jest test suite...",
        f"FAIL {f.replace('.py', '.test.js')}",
        f"  ✕ {n.replace('_', ' ')} ({random.randint(200, 5000)}ms)",
        f"  ● {n.replace('_', ' ')}",
        f"",
        f"    Expected: 200",
        f"    Received: {random.choice(HTTP_CODES)}",
        f"",
        f"    > {random.randint(30, 80)} | expect(response.status).toBe(200);",
        f"",
        f"Tests: {random.randint(1, 5)} failed, {random.randint(10, 50)} passed",
        f"Test Suites: 1 failed, {random.randint(3, 10)} passed",
    ],
    lambda n, f: [
        f"pytest --cov=src --cov-fail-under=80 tests/",
        f"collected {random.randint(50, 120)} items",
        f"",
        f"FAILED {f}::{n} - TimeoutError: test exceeded {random.randint(5, 30)}s limit",
        f"",
        f"---------- coverage: platform linux, python 3.11 -----------",
        f"Name                      Stmts   Miss  Cover",
        f"---------------------------------------------------------",
        f"src/api.py                  120     48    60%",
        f"src/models.py                95     31    67%",
        f"---------------------------------------------------------",
        f"TOTAL                       215     79    {random.randint(55, 79)}%",
        f"FAIL Required test coverage of 80% not reached. Total coverage: {random.randint(55, 79)}%",
        f"{'=' * 50}",
        f"{random.randint(1, 4)} failed, {random.randint(20, 60)} passed in {random.uniform(5.0, 60.0):.2f}s",
    ],
]


def generate_test_failure_log() -> str:
    """Generate a realistic test failure log entry."""
    lines = _pipeline_header("test_failure")
    t = _ts(1)
    lines.append(f"[{t}] Stage: test")
    lines.append(f"[{t}] Running: pytest tests/ -v --cov=src")

    n = random.choice(TEST_NAMES)
    f = random.choice(TEST_FILES)
    tmpl = random.choice(_PYTEST_FAILURES)
    lines.extend(tmpl(n, f))

    t2 = _ts(3)
    lines.append(f"[{t2}] ERROR: Stage 'test' failed")
    lines.append(f"[{t2}] Pipeline run aborted. Check test results above.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TIMEOUT GENERATORS
# ---------------------------------------------------------------------------

TIMEOUT_MINUTES = [10, 15, 20, 30, 45, 60]
HUNG_TASKS = [
    "waiting for database migration to complete",
    "running end-to-end browser tests",
    "building Docker image layers",
    "running npm ci with 1200 packages",
    "waiting for Kubernetes rollout",
    "running database seeding script",
    "waiting for external API health check",
]

_TIMEOUT_SCENARIOS = [
    lambda mins: [
        f"##[error]The job running on runner GitHub Actions 3 has exceeded the maximum execution time of {mins} minutes.",
        f"##[error]The runner has received a shutdown signal.",
        f"##[warning]Steps that are still running will be cancelled.",
        f"##[error]Process '/usr/bin/bash' exited with code 1 and signal 15 (SIGTERM).",
        f"Error: The operation was canceled.",
    ],
    lambda mins: [
        f"CI_JOB_STATUS: running",
        f"Job exceeded {mins}m time limit",
        f"Sending SIGTERM to process group {random.randint(1000, 9999)}",
        f"Process did not exit within 10 seconds after SIGTERM",
        f"Sending SIGKILL",
        f"ERROR: Job failed: execution took longer than {mins * 60} seconds",
    ],
    lambda mins: [
        f"[Pipeline] timeout",
        f"Timeout set to expire in {mins} min {random.randint(0,59)} sec",
        f"FlowInterruptedException: org.jenkinsci.plugins.workflow.steps.FlowInterruptedException",
        f"    at org.jenkinsci.plugins.workflow.steps.TimeoutStep$Callback.timeoutReached",
        f"Finished: ABORTED",
    ],
    lambda mins: [
        f"agent lost connection after {mins} minutes",
        f"Retrying connection to runner... (1/3)",
        f"Retrying connection to runner... (2/3)",
        f"Retrying connection to runner... (3/3)",
        f"ERROR: No response from runner {random.choice(RUNNER_IDS)} after {mins * 60}s",
        f"Job marked as failed due to runner disconnection.",
    ],
    lambda mins: [
        f"Request timeout after {mins * 60}s waiting for health endpoint /health",
        f"curl: (28) Operation timed out after {mins * 60000} milliseconds",
        f"Health check failed after {random.randint(3, 10)} attempts",
        f"Service did not become ready in time",
        f"Deployment timeout: rolling update exceeded {mins}m limit",
    ],
]


def generate_timeout_log() -> str:
    """Generate a realistic timeout log entry."""
    lines = _pipeline_header("timeout")
    t = _ts(0)
    mins = random.choice(TIMEOUT_MINUTES)
    task = random.choice(HUNG_TASKS)
    lines.append(f"[{t}] Stage: {task}")
    lines.append(f"[{t}] Timeout limit: {mins}m")

    # Add some progress lines before timeout
    for i in range(random.randint(2, 5)):
        t_mid = _ts(random.randint(1, mins - 1))
        lines.append(f"[{t_mid}] Still {task}... ({i * 20}% complete)")

    t2 = _ts(mins)
    lines.append(f"[{t2}] TIMEOUT REACHED after {mins} minutes")
    tmpl = random.choice(_TIMEOUT_SCENARIOS)
    lines.extend(tmpl(mins))

    t3 = _ts(mins + 1)
    lines.append(f"[{t3}] Pipeline run TIMED OUT and was terminated.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CONFIG ERROR GENERATORS
# ---------------------------------------------------------------------------

ENV_VARS = ["DATABASE_URL", "REDIS_URL", "SECRET_KEY", "AWS_ACCESS_KEY_ID",
            "STRIPE_API_KEY", "SENDGRID_API_KEY", "JWT_SECRET", "POSTGRES_PASSWORD",
            "S3_BUCKET_NAME", "OAUTH_CLIENT_SECRET"]
CONFIG_FILES = ["/app/config.yml", "/etc/app/settings.yaml", "config/production.json",
                ".env.production", "/run/secrets/db-password", "k8s/configmap.yaml"]
BAD_PATHS = ["/app/data", "/var/log/app", "/tmp/uploads", "/mnt/storage"]

_CONFIG_ERRORS = [
    lambda v: [
        f"Loading application configuration...",
        f"Traceback (most recent call last):",
        f"  File \"/app/config.py\", line {random.randint(20, 80)}, in load_config",
        f"    {v} = os.environ['{v}']",
        f"KeyError: '{v}'",
        f"",
        f"Environment variable '{v}' is not set.",
        f"Please configure required environment variables before deployment.",
    ],
    lambda v: [
        f"yaml.scanner.ScannerError: mapping values are not allowed here",
        f"  in \"config/pipeline.yml\", line {random.randint(5, 50)}, column {random.randint(3, 20)}",
        f"",
        f"  {random.randint(10,40)} |   {v.lower()}: ${{{{ secrets.{v} }}}}:wrong",
        f"  {' ' * 5}| {'^ ' * 8}",
        f"",
        f"ERROR: Invalid YAML syntax in pipeline configuration.",
        f"Hint: Check indentation and special characters.",
    ],
    lambda v: [
        f"Mounting secret '{v.lower()}-secret'...",
        f"Error from server (Forbidden): secrets \"{v.lower()}-secret\" is forbidden:",
        f"User \"system:serviceaccount:default:app-sa\" cannot get resource \"secrets\"",
        f"in API group \"\" in the namespace \"production\"",
        f"",
        f"PermissionError: [Errno 13] Permission denied: '/run/secrets/{v.lower()}'",
        f"ERROR: Cannot read required secret. Check RBAC configuration.",
    ],
    lambda v: [
        f"Reading config from /app/config.yml...",
        f"FileNotFoundError: [Errno 2] No such file or directory: '/app/config.yml'",
        f"",
        f"  File \"/app/startup.py\", line {random.randint(10, 40)}, in main",
        f"    config = load_yaml('/app/config.yml')",
        f"",
        f"Please ensure configuration file is mounted at /app/config.yml",
        f"Hint: Add to your Dockerfile: COPY config/production.yml /app/config.yml",
    ],
    lambda v: [
        f"Validating pipeline configuration...",
        f"jsonschema.exceptions.ValidationError: '{random.choice(['null', 'undefined', ''])}' is not of type 'string'",
        f"",
        f"Failed validating 'type' in schema['properties']['{v.lower()}']:",
        f"    {{'type': 'string', 'minLength': 1}}",
        f"",
        f"On instance['{v.lower()}']:",
        f"    None",
        f"",
        f"ERROR: Configuration validation failed. '{v}' must be a non-empty string.",
    ],
]


def generate_config_error_log() -> str:
    """Generate a realistic configuration error log entry."""
    lines = _pipeline_header("config_error")
    t = _ts(0)
    v = random.choice(ENV_VARS)
    lines.append(f"[{t}] Stage: configure")
    lines.append(f"[{t}] Loading environment: production")

    tmpl = random.choice(_CONFIG_ERRORS)
    lines.extend(tmpl(v))

    t2 = _ts(1)
    lines.append(f"[{t2}] ERROR: Configuration stage failed. Pipeline cannot continue.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# INFRA FAILURE GENERATORS
# ---------------------------------------------------------------------------

K8S_PODS = [f"app-deployment-{random.randint(1,9)}{random.randint(10,99)}" for _ in range(20)]
DOCKER_IMAGES = ["myapp:latest", "nginx:1.25", "postgres:15", "redis:7-alpine", "myapp:v2.1.0"]
NAMESPACES = ["production", "staging", "default", "monitoring", "ingress-nginx"]

_INFRA_ERRORS = [
    lambda pod: [
        f"kubectl get pods -n {random.choice(NAMESPACES)}",
        f"NAME                           READY   STATUS             RESTARTS   AGE",
        f"{pod}-abc12   0/1     OOMKilled          {random.randint(1,8)}          {random.randint(1,60)}m",
        f"",
        f"Warning: OOMKilled: container 'app' exceeded memory limit of {random.choice(['256Mi','512Mi','1Gi'])}",
        f"Last State: Terminated (OOMKilled) at {_ts(-1)}",
        f"Pod {pod}-abc12 is not healthy. Deployment rollout failed.",
    ],
    lambda pod: [
        f"Waiting for deployment 'app-deployment' rollout to finish...",
        f"  Waiting for pod {pod}-xyz99 to be ready...",
        f"",
        f"Warning  BackOff  2m    kubelet  Back-off restarting failed container",
        f"Normal   Pulled   2m    kubelet  Successfully pulled image {random.choice(DOCKER_IMAGES)}",
        f"Warning  Failed   90s   kubelet  Error: failed to start container",
        f"",
        f"Deployment 'app-deployment' rollout status: CrashLoopBackOff",
        f"ERROR: Deployment did not stabilize within 5 minutes.",
    ],
    lambda pod: [
        f"docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
        f"Is the docker daemon running?",
        f"",
        f"Error response from daemon: OCI runtime create failed:",
        f"container_linux.go:370: starting container process caused: process_linux.go:459:",
        f"container init caused: rootfs_linux.go:76: mounting ... caused:",
        f"mount through procfd: not a directory: unknown",
        f"ERROR: docker build exited with code 125",
    ],
    lambda pod: [
        f"Checking disk space on build host...",
        f"Filesystem      Size  Used Avail Use% Mounted on",
        f"/dev/sda1        50G   50G     0 100% /",
        f"/dev/sdb1       100G   98G   2.0G  98% /var/lib/docker",
        f"",
        f"OSError: [Errno 28] No space left on device: '/var/lib/docker/overlay2/...'",
        f"docker: failed to copy files: write /var/lib/docker/tmp/...: no space left on device.",
        f"ERROR: Build failed due to insufficient disk space.",
    ],
    lambda pod: [
        f"Pulling image {random.choice(DOCKER_IMAGES)}...",
        f"ErrImagePull: rpc error: code = Unknown desc = failed to pull and unpack image",
        f'"{random.choice(DOCKER_IMAGES)}": failed to resolve reference',
        f'"{random.choice(DOCKER_IMAGES)}": unexpected status code 403 Forbidden',
        f"",
        f"Warning  Failed  30s   kubelet  Failed to pull image: ErrImagePull",
        f"Warning  Failed  30s   kubelet  Error: ErrImagePull",
        f"Warning  BackOff 15s   kubelet  Back-off pulling image '{random.choice(DOCKER_IMAGES)}'",
        f"Pod status: ImagePullBackOff",
    ],
    lambda pod: [
        f"kubectl describe pod {pod}-def45 -n {random.choice(NAMESPACES)}",
        f"Events:",
        f"  Type     Reason          Age   From               Message",
        f"  ----     ------          ---   ----               -------",
        f"  Warning  NetworkNotReady 5m    node-controller    network is not ready: container runtime network not ready",
        f"  Warning  FailedMount     3m    kubelet            MountVolume.SetUp failed for volume 'config-vol'",
        f"  Normal   Killing         1m    kubelet            Stopping container app",
        f"",
        f"dial tcp: lookup db.internal: no such host",
        f"ERROR: Pod failed to start due to network/storage issues.",
    ],
]


def generate_infra_failure_log() -> str:
    """Generate a realistic infrastructure failure log entry."""
    lines = _pipeline_header("infra_failure")
    t = _ts(0)
    pod = random.choice(K8S_PODS)
    lines.append(f"[{t}] Stage: deploy")
    lines.append(f"[{t}] Target: Kubernetes cluster / namespace: {random.choice(NAMESPACES)}")

    tmpl = random.choice(_INFRA_ERRORS)
    lines.extend(tmpl(pod))

    t2 = _ts(5)
    lines.append(f"[{t2}] ERROR: Deployment failed. Infrastructure issue detected.")
    lines.append(f"[{t2}] Rollback initiated to previous stable version.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DRIFT LOG GENERATORS (Kubernetes-heavy vocabulary)
# ---------------------------------------------------------------------------

K8S_CONTROLLERS = ["StatefulSet", "DaemonSet", "ReplicaSet", "CronJob", "Job"]
K8S_CONDITIONS = ["PodScheduled", "ContainersReady", "Initialized", "Ready"]

_DRIFT_SCENARIOS = [
    lambda pod: [
        f"[{_ts(0)}] Liveness probe failed for pod {pod}",
        f"GET /healthz: dial tcp 10.0.{random.randint(0,255)}.{random.randint(1,254)}:{random.randint(8000,9000)}: connect: connection refused",
        f"Liveness probe failed: HTTP probe failed with statuscode: {random.choice([500, 503, 000])}",
        f"Container will be restarted due to failed liveness probe.",
        f"Warning  Unhealthy  {random.randint(1,10)}m   kubelet  Liveness probe failed",
        f"Normal   Killing    {random.randint(1,5)}m   kubelet  Container {pod} failed liveness probe, will be restarted",
    ],
    lambda pod: [
        f"[{_ts(0)}] Readiness probe timeout for pod {pod}",
        f"Readiness probe failed: Get 'http://:{random.randint(8000,9000)}/ready': dial tcp: connection refused",
        f"Pod readiness probe timeout after {random.randint(10,60)}s",
        f"Warning  Unhealthy  {random.randint(1,10)}m   kubelet  Readiness probe failed: HTTP probe failed",
        f"Pod {pod} is not ready. Endpoints will not receive traffic.",
        f"Service {pod.rsplit('-',1)[0]}-svc has 0/{random.randint(2,5)} ready endpoints",
    ],
    lambda pod: [
        f"[{_ts(0)}] Image pull failed: ImagePullBackOff",
        f"Warning  Failed      {random.randint(1,5)}m   kubelet  Failed to pull image 'registry.internal/{pod}:v{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)}'",
        f"Warning  Failed      {random.randint(1,5)}m   kubelet  Error: ImagePullBackOff",
        f"Warning  BackOff     {random.randint(30,120)}s  kubelet  Back-off pulling image",
        f"Error: ImagePullBackOff - Back-off pulling image, will retry later",
        f"Kubernetes is waiting for image to become available before scheduling pod.",
    ],
    lambda pod: [
        f"[{_ts(0)}] Pod scheduling failed: PodUnschedulable",
        f"0/{random.randint(3,10)} nodes are available:",
        f"  {random.randint(1,3)} Insufficient cpu.",
        f"  {random.randint(1,4)} Insufficient memory.",
        f"  1 node(s) had taint {{node.kubernetes.io/not-ready:NoSchedule}}",
        f"preemption: 0/3 nodes are available: 3 No preemption victims found.",
        f"Warning  FailedScheduling  {random.randint(1,10)}m  default-scheduler  0/{random.randint(3,8)} nodes are available",
        f"Pod status: Pending | Reason: Unschedulable",
    ],
    lambda pod: [
        f"[{_ts(0)}] Volume mount failure: FailedMount",
        f"Warning  FailedMount  {random.randint(1,10)}m  kubelet  MountVolume.SetUp failed for volume 'persistent-storage':",
        f"  rpc error: code = Internal desc = failed to stage volume: mount failed:",
        f"  mounting device /dev/sdb at /var/lib/kubelet/pods/.../volumes/... failed:",
        f"  exit status 32",
        f"Warning  FailedMount  {random.randint(1,5)}m  kubelet  Unable to attach or mount volumes: unmounted volumes=[config-vol]",
        f"Pod cannot start: required volume mounts are unavailable.",
    ],
    lambda pod: [
        f"[{_ts(0)}] HorizontalPodAutoscaler scaling event",
        f"HPA {pod.rsplit('-',1)[0]}-hpa unable to compute replica count:",
        f"  failed to get metrics: unable to fetch metrics from resource metrics API:",
        f"  the server is currently unable to handle the request (get pods.metrics.k8s.io)",
        f"Warning  FailedGetScale  {random.randint(1,5)}m  horizontal-pod-autoscaler  failed to get cpu utilization",
        f"PodDisruptionBudget '{pod.rsplit('-',1)[0]}-pdb' minimum available replicas not met",
        f"Deployment paused due to PDB constraint violation.",
    ],
]


def generate_drift_log() -> str:
    """Generate a drifted log with Kubernetes-heavy terminology."""
    pod = f"{random.choice(['web', 'api', 'worker', 'cache', 'db'])}-{random.choice(['deployment', 'statefulset', 'daemonset'])}-{random.randint(10000, 99999)}"
    t = _ts(0)
    ns = random.choice(NAMESPACES)
    lines = [
        f"[{t}] Kubernetes event detected in namespace: {ns}",
        f"[{t}] Controller: {random.choice(K8S_CONTROLLERS)} | Pod: {pod}",
    ]
    tmpl = random.choice(_DRIFT_SCENARIOS)
    lines.extend(tmpl(pod))

    t2 = _ts(1)
    controller = random.choice(K8S_CONTROLLERS)
    lines.append(f"[{t2}] {controller} rollout status: degraded")
    lines.append(f"[{t2}] Condition '{random.choice(K8S_CONDITIONS)}': False")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN GENERATION LOGIC
# ---------------------------------------------------------------------------

GENERATORS = {
    "build_failure": generate_build_failure_log,
    "test_failure": generate_test_failure_log,
    "timeout": generate_timeout_log,
    "config_error": generate_config_error_log,
    "infra_failure": generate_infra_failure_log,
}

SAMPLES_PER_CLASS = 120
DRIFT_SAMPLES = 150


def _generate_timestamps(n: int, days_back: int = 30) -> list[str]:
    """Generate n realistic timestamps spread over the last `days_back` days."""
    now = datetime.now()
    timestamps = []
    for _ in range(n):
        delta = timedelta(
            days=random.uniform(0, days_back),
            hours=random.uniform(0, 24),
            minutes=random.uniform(0, 60),
        )
        timestamps.append((now - delta).strftime("%Y-%m-%d %H:%M:%S"))
    return sorted(timestamps)


def generate_training_data() -> pd.DataFrame:
    """Generate 600 labeled training log samples (120 per class).

    Returns:
        DataFrame with columns: [id, log_text, label, timestamp]
    """
    rows = []
    timestamps = _generate_timestamps(SAMPLES_PER_CLASS * len(GENERATORS), days_back=30)
    ts_idx = 0

    for label, generator in GENERATORS.items():
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Generating {SAMPLES_PER_CLASS} samples: {label}")
        for i in range(SAMPLES_PER_CLASS):
            log_text = generator()
            rows.append({
                "id": f"{label}_{i:04d}",
                "log_text": log_text,
                "label": label,
                "timestamp": timestamps[ts_idx],
            })
            ts_idx += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_drift_data() -> pd.DataFrame:
    """Generate 150 drifted log samples with Kubernetes-heavy terminology.

    Returns:
        DataFrame with columns: [id, log_text, label, timestamp]
    """
    rows = []
    timestamps = _generate_timestamps(DRIFT_SAMPLES, days_back=7)
    labels_pool = list(GENERATORS.keys())

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Generating {DRIFT_SAMPLES} drift samples")
    for i in range(DRIFT_SAMPLES):
        log_text = generate_drift_log()
        label = random.choice(labels_pool)
        rows.append({
            "id": f"drift_{i:04d}",
            "log_text": log_text,
            "label": label,
            "timestamp": timestamps[i],
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Entry point: generate and save training and drift datasets."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === DevOps Log Data Generator ===")

    train_path = DATA_RAW / "logs.csv"
    drift_path = DATA_DRIFT / "logs_drift.csv"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating training data ({SAMPLES_PER_CLASS * len(GENERATORS)} samples)...")
    train_df = generate_training_data()
    train_df.to_csv(train_path, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training data saved: {train_path}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Label distribution:")
    for label, count in train_df["label"].value_counts().items():
        print(f"  {label}: {count} samples")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating drift data ({DRIFT_SAMPLES} samples)...")
    drift_df = generate_drift_data()
    drift_df.to_csv(drift_path, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Drift data saved: {drift_path}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done. Total samples: {len(train_df)} training, {len(drift_df)} drift.")


if __name__ == "__main__":
    main()
