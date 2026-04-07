import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import shutil
import urllib.request
import urllib.parse
import urllib.error
import binascii # Base64 에러 처리를 위해 import
import time
import copy
from functools import lru_cache

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_ROOT = os.getenv("JOB_TEMP_ROOT", "/tmp/generate_video")
server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')

DEFAULT_NEGATIVE_PROMPT = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
DEFAULT_LENGTH = 17
DEFAULT_STEPS = 8
DEFAULT_SEED = 42
DEFAULT_CFG = 2.0
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 832
DEFAULT_CONTEXT_OVERLAP = 16
MAX_LORA_PAIRS = 4
HTTP_TIMEOUT_SECONDS = int(os.getenv("COMFY_HTTP_TIMEOUT_SECONDS", "30"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("INPUT_DOWNLOAD_TIMEOUT_SECONDS", "120"))
WS_RECV_TIMEOUT_SECONDS = int(os.getenv("COMFY_WS_TIMEOUT_SECONDS", "1800"))
HISTORY_RETRY_COUNT = int(os.getenv("COMFY_HISTORY_RETRY_COUNT", "10"))
HISTORY_RETRY_DELAY_SECONDS = float(os.getenv("COMFY_HISTORY_RETRY_DELAY_SECONDS", "1"))
BLOCKSWAP_NODE_ID = "525"
BLOCKSWAP_SETTER_NODE_ID = "555"
LORA_SETTER_NODE_ID = "556"
SINGLE_WORKFLOW_PATH = os.path.join(BASE_DIR, "new_Wan22_api.json")
FLF2V_WORKFLOW_PATH = os.path.join(BASE_DIR, "new_Wan22_flf2v_api.json")
DEFAULT_IMAGE_PATH = os.path.join(BASE_DIR, "example_image.png")
COMMON_MUTATED_NODE_IDS = {"122", "135", "220", "235", "236", "244", "279", "498", "541", "556", "569"}
FLF2V_MUTATED_NODE_IDS = {"617"}


def clamp_int(value, minimum, maximum=None):
    try:
        numeric_value = int(value)
    except Exception:
        raise Exception(f"정수 값이 아닙니다: {value}")

    if maximum is not None:
        numeric_value = min(numeric_value, maximum)

    return max(minimum, numeric_value)


def to_nearest_multiple_of_16(value):
    """주어진 값을 가장 가까운 16의 배수로 보정, 최소 16 보장"""
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height 값이 숫자가 아닙니다: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted


def parse_float(value, field_name):
    try:
        return float(value)
    except Exception:
        raise Exception(f"{field_name} 값이 숫자가 아닙니다: {value}")


def is_value_provided(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def resolve_input_source(job_input, field_prefix):
    candidates = []
    for input_type, field_name in (
        ("path", f"{field_prefix}_path"),
        ("url", f"{field_prefix}_url"),
        ("base64", f"{field_prefix}_base64"),
    ):
        value = job_input.get(field_name)
        if is_value_provided(value):
            candidates.append((input_type, value, field_name))

    if len(candidates) > 1:
        provided_fields = ", ".join(field_name for _, _, field_name in candidates)
        raise Exception(f"{field_prefix} 입력은 하나만 지정할 수 있습니다: {provided_fields}")

    if not candidates:
        return None, None

    input_type, value, _ = candidates[0]
    return input_type, value


def make_temp_dir(task_id):
    safe_task_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(task_id))
    if not safe_task_id:
        safe_task_id = str(uuid.uuid4())
    temp_dir = os.path.join(TEMP_ROOT, safe_task_id)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def normalize_base64_data(base64_data):
    if not isinstance(base64_data, str):
        raise Exception("Base64 입력은 문자열이어야 합니다.")

    normalized_data = base64_data.strip()
    if normalized_data.startswith("data:"):
        _, _, normalized_data = normalized_data.partition(",")

    normalized_data = "".join(normalized_data.split())
    if not normalized_data:
        raise Exception("Base64 입력이 비어 있습니다.")

    return normalized_data


def process_input(input_data, temp_dir, output_filename, input_type):
    """입력 데이터를 처리하여 파일 경로를 반환하는 함수"""
    if input_type == "path":
        # 경로인 경우 그대로 반환
        resolved_path = os.path.abspath(os.path.expanduser(str(input_data)))
        if not os.path.isfile(resolved_path):
            raise Exception(f"입력 파일을 찾을 수 없습니다: {resolved_path}")
        logger.info(f"📁 경로 입력 처리: {resolved_path}")
        return resolved_path
    elif input_type == "url":
        # URL인 경우 다운로드
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        # Base64인 경우 디코딩하여 저장
        logger.info(f"🔢 Base64 입력 처리")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"지원하지 않는 입력 타입: {input_type}")

        
def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    try:
        parsed_url = urllib.parse.urlparse(str(url))
        if parsed_url.scheme not in {"http", "https"}:
            raise Exception(f"지원하지 않는 URL 스킴입니다: {parsed_url.scheme}")

        request = urllib.request.Request(str(url), headers={"User-Agent": "generate_video-worker/1.0"})
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(output_path, 'wb') as output_file:
            shutil.copyfileobj(response, output_file)

        logger.info(f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}")
        return output_path
    except urllib.error.URLError as exc:
        logger.error(f"❌ URL 다운로드 실패: {exc}")
        raise Exception(f"URL 다운로드 실패: {exc}")
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise Exception(f"다운로드 중 오류 발생: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        # Base64 문자열 디코딩
        decoded_data = base64.b64decode(normalize_base64_data(base64_data), validate=True)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 파일로 저장
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        
        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise Exception(f"Base64 디코딩 실패: {e}")
    
def queue_prompt(prompt, request_client_id):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": request_client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS).read())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.error(f"ComfyUI rejected the prompt: {error_body}")
        raise

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}", timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def get_history_with_retry(prompt_id):
    for attempt in range(HISTORY_RETRY_COUNT):
        history = get_history(prompt_id)
        prompt_history = history.get(prompt_id)
        if prompt_history is not None:
            return prompt_history

        logger.warning(f"History for prompt {prompt_id} is not ready yet ({attempt + 1}/{HISTORY_RETRY_COUNT}).")
        time.sleep(HISTORY_RETRY_DELAY_SECONDS)

    raise Exception(f"ComfyUI history를 가져오지 못했습니다: {prompt_id}")


def get_available_node_types():
    url = f"http://{server_address}:8188/object_info"
    logger.info(f"Fetching ComfyUI object info from: {url}")
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
            object_info = json.loads(response.read())
    except Exception as exc:
        logger.warning(f"ComfyUI node inventory could not be fetched. Skipping optional-node validation: {exc}")
        return None

    if not isinstance(object_info, dict):
        logger.warning("Unexpected /object_info response from ComfyUI. Skipping optional-node validation.")
        return None

    return set(object_info.keys())


def bypass_blockswap(prompt):
    blockswap_setter = prompt.get(BLOCKSWAP_SETTER_NODE_ID, {})
    lora_setter = prompt.get(LORA_SETTER_NODE_ID, {})
    upstream_model = blockswap_setter.get("inputs", {}).get("model")

    if upstream_model is None or "inputs" not in lora_setter:
        logger.warning("BlockSwap fallback was requested, but the workflow shape does not match the expected WanVideo graph.")
        return

    lora_setter["inputs"]["model"] = upstream_model
    prompt.pop(BLOCKSWAP_NODE_ID, None)
    prompt.pop(BLOCKSWAP_SETTER_NODE_ID, None)
    logger.warning("WanVideo BlockSwap nodes are unavailable in ComfyUI. Rewiring the workflow to use the base model directly.")


def validate_prompt_node_types(prompt):
    available_node_types = get_available_node_types()
    if not available_node_types:
        return

    missing_node_types = {
        node.get("class_type")
        for node in prompt.values()
        if isinstance(node, dict) and node.get("class_type") not in available_node_types
    }
    missing_node_types.discard(None)

    optional_blockswap_classes = {"WanVideoEnhancedBlockSwap", "WanVideoSetBlockSwap"}
    if missing_node_types & optional_blockswap_classes:
        bypass_blockswap(prompt)
        missing_node_types -= optional_blockswap_classes
        missing_node_types = {
            node.get("class_type")
            for node in prompt.values()
            if isinstance(node, dict) and node.get("class_type") not in available_node_types
        }
        missing_node_types.discard(None)

    if missing_node_types:
        missing_list = ", ".join(sorted(missing_node_types))
        raise Exception(f"ComfyUI is missing required workflow node types: {missing_list}")


def validate_workflow_structure(prompt, workflow_file, required_node_ids):
    missing_required_node_ids = sorted(required_node_ids - set(prompt.keys()))
    if missing_required_node_ids:
        missing_ids = ", ".join(missing_required_node_ids)
        raise Exception(f"워크플로우 파일에 필요한 노드가 없습니다 ({workflow_file}): {missing_ids}")

    dangling_references = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue

        for input_name, input_value in node.get("inputs", {}).items():
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
                and isinstance(input_value[1], int)
                and input_value[0] not in prompt
            ):
                dangling_references.append((node_id, input_name, input_value[0]))

    if dangling_references:
        node_id, input_name, missing_reference = dangling_references[0]
        raise Exception(
            f"워크플로우 참조가 잘못되었습니다 ({workflow_file}): "
            f"node {node_id} input '{input_name}' -> missing node {missing_reference}"
        )


def normalize_lora_pairs(raw_lora_pairs):
    if raw_lora_pairs is None:
        return []
    if not isinstance(raw_lora_pairs, list):
        raise Exception("lora_pairs는 배열이어야 합니다.")

    normalized_pairs = []
    for index, lora_pair in enumerate(raw_lora_pairs[:MAX_LORA_PAIRS]):
        if not isinstance(lora_pair, dict):
            raise Exception(f"lora_pairs[{index}]는 객체여야 합니다.")
        normalized_pairs.append(lora_pair)

    if len(raw_lora_pairs) > MAX_LORA_PAIRS:
        logger.warning(f"LoRA 개수가 {len(raw_lora_pairs)}개입니다. 최대 4개까지만 지원됩니다. 처음 4개만 사용합니다.")

    return normalized_pairs


def prepare_job_config(job_input, task_id):
    if not isinstance(job_input, dict):
        raise Exception("job.input은 JSON 객체여야 합니다.")

    temp_dir = make_temp_dir(task_id)

    image_input_type, image_input_value = resolve_input_source(job_input, "image")
    if image_input_type:
        image_path = process_input(image_input_value, temp_dir, "input_image.jpg", image_input_type)
    else:
        image_path = DEFAULT_IMAGE_PATH
        if not os.path.exists(image_path):
            raise Exception(f"기본 이미지 파일을 찾을 수 없습니다: {image_path}")
        logger.info(f"기본 이미지 파일을 사용합니다: {image_path}")

    end_image_input_type, end_image_input_value = resolve_input_source(job_input, "end_image")
    end_image_path_local = None
    if end_image_input_type:
        end_image_path_local = process_input(end_image_input_value, temp_dir, "end_image.jpg", end_image_input_type)

    prompt_text = str(job_input.get("prompt", "")).strip()
    if not prompt_text:
        raise Exception("prompt는 필수 입력입니다.")

    length = clamp_int(job_input.get("length", DEFAULT_LENGTH), 1)

    return {
        "temp_dir": temp_dir,
        "image_path": image_path,
        "end_image_path_local": end_image_path_local,
        "prompt_text": prompt_text,
        "negative_prompt": (
            DEFAULT_NEGATIVE_PROMPT
            if job_input.get("negative_prompt") is None
            else str(job_input.get("negative_prompt"))
        ),
        "seed": clamp_int(job_input.get("seed", DEFAULT_SEED), 0),
        "cfg": parse_float(job_input.get("cfg", DEFAULT_CFG), "cfg"),
        "width": to_nearest_multiple_of_16(job_input.get("width", DEFAULT_WIDTH)),
        "height": to_nearest_multiple_of_16(job_input.get("height", DEFAULT_HEIGHT)),
        "length": length,
        "steps": clamp_int(job_input.get("steps", DEFAULT_STEPS), 1),
        "context_overlap": clamp_int(
            job_input.get("context_overlap", DEFAULT_CONTEXT_OVERLAP),
            0,
            max(length - 1, 0),
        ),
        "lora_pairs": normalize_lora_pairs(job_input.get("lora_pairs", [])),
    }


def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def format_execution_error(error_data):
    if not isinstance(error_data, dict):
        return f"ComfyUI execution failed: {error_data}"

    node_id = error_data.get("node_id") or error_data.get("node")
    exception_type = error_data.get("exception_type")
    exception_message = error_data.get("exception_message") or error_data.get("error")

    error_parts = ["ComfyUI execution failed"]
    if node_id is not None:
        error_parts.append(f"at node {node_id}")
    if exception_type:
        error_parts.append(f"({exception_type})")
    if exception_message:
        error_parts.append(f": {exception_message}")

    formatted_message = " ".join(error_parts)
    traceback_text = error_data.get("traceback")
    if traceback_text:
        formatted_message = f"{formatted_message}\n{traceback_text}"

    return formatted_message


def maybe_upload_video(job_id, video_path):
    required_env_vars = (
        "BUCKET_ENDPOINT_URL",
        "BUCKET_ACCESS_KEY_ID",
        "BUCKET_SECRET_ACCESS_KEY",
    )
    if not all(os.environ.get(var_name) for var_name in required_env_vars):
        return None

    upload_file = getattr(rp_upload, "upload_file_to_bucket", None)
    if upload_file is None:
        logger.warning("rp_upload.upload_file_to_bucket is unavailable. Falling back to base64 output.")
        return None

    try:
        return upload_file(
            file_name=os.path.basename(video_path),
            file_location=video_path,
            prefix=job_id,
        )
    except Exception as exc:
        logger.warning(f"비디오 업로드 실패, base64 결과로 fallback 합니다: {exc}")
        return None


def get_videos(ws, prompt, job_id, request_client_id):
    queue_response = queue_prompt(prompt, request_client_id)
    prompt_id = queue_response.get('prompt_id')
    if not prompt_id:
        raise Exception(f"ComfyUI prompt_id를 받지 못했습니다: {queue_response}")

    output_videos = {}
    while True:
        try:
            out = ws.recv()
        except websocket.WebSocketTimeoutException as exc:
            raise Exception(f"ComfyUI 응답 대기 시간이 초과되었습니다: {exc}")
        except websocket.WebSocketConnectionClosedException as exc:
            raise Exception(f"ComfyUI 웹소켓 연결이 종료되었습니다: {exc}")

        if isinstance(out, str):
            try:
                message = json.loads(out)
            except json.JSONDecodeError:
                logger.warning(f"ComfyUI에서 JSON이 아닌 웹소켓 메시지를 받았습니다: {out}")
                continue
            message_type = message.get('type')
            data = message.get('data', {})
            if message_type == 'execution_error':
                raise Exception(format_execution_error(data))
            if message_type == 'execution_interrupted':
                raise Exception("ComfyUI 실행이 중단되었습니다.")
            if message_type == 'executing':
                if data.get('node') is None and data.get('prompt_id') == prompt_id:
                    break
        else:
            continue

    history = get_history_with_retry(prompt_id)
    outputs = history.get('outputs')
    if not outputs:
        raise Exception(f"ComfyUI history에 outputs가 없습니다: {history}")

    for node_id in outputs:
        node_output = outputs[node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                video_url = maybe_upload_video(job_id, video['fullpath'])
                if video_url:
                    videos_output.append({"video_url": video_url})
                    continue

                with open(video['fullpath'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                videos_output.append({"video": video_data})
        output_videos[node_id] = videos_output

    return output_videos

@lru_cache(maxsize=2)
def load_workflow_template(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

def load_workflow(workflow_path):
    return copy.deepcopy(load_workflow_template(workflow_path))

def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = job.get("id", f"task_{uuid.uuid4()}")
    request_client_id = str(uuid.uuid4())
    ws = None
    temp_dir = None

    try:
        job_config = prepare_job_config(job_input, task_id)
        temp_dir = job_config["temp_dir"]
        image_path = job_config["image_path"]
        end_image_path_local = job_config["end_image_path_local"]
        lora_pairs = job_config["lora_pairs"]
        lora_count = len(lora_pairs)

        workflow_file = FLF2V_WORKFLOW_PATH if end_image_path_local else SINGLE_WORKFLOW_PATH
        logger.info(f"Using {'FLF2V' if end_image_path_local else 'single'} workflow with {lora_count} LoRA pairs")
        prompt = load_workflow(workflow_file)

        required_node_ids = set(COMMON_MUTATED_NODE_IDS)
        if end_image_path_local:
            required_node_ids |= FLF2V_MUTATED_NODE_IDS
        validate_workflow_structure(prompt, workflow_file, required_node_ids)

        prompt["244"]["inputs"]["image"] = image_path
        prompt["541"]["inputs"]["num_frames"] = job_config["length"]
        prompt["135"]["inputs"]["positive_prompt"] = job_config["prompt_text"]
        prompt["135"]["inputs"]["negative_prompt"] = job_config["negative_prompt"]
        prompt["220"]["inputs"]["seed"] = job_config["seed"]
        prompt["220"]["inputs"]["cfg"] = job_config["cfg"]

        original_width = job_input.get("width", DEFAULT_WIDTH)
        original_height = job_input.get("height", DEFAULT_HEIGHT)
        if job_config["width"] != int(float(original_width)):
            logger.info(f"Width adjusted to nearest multiple of 16: {original_width} -> {job_config['width']}")
        if job_config["height"] != int(float(original_height)):
            logger.info(f"Height adjusted to nearest multiple of 16: {original_height} -> {job_config['height']}")

        prompt["235"]["inputs"]["value"] = job_config["width"]
        prompt["236"]["inputs"]["value"] = job_config["height"]
        prompt["498"]["inputs"]["context_overlap"] = job_config["context_overlap"]
        prompt["498"]["inputs"]["context_frames"] = job_config["length"]
        prompt["569"]["inputs"]["value"] = job_config["steps"]
        logger.info(f"Steps set to: {job_config['steps']}")

        if end_image_path_local:
            prompt["617"]["inputs"]["image"] = end_image_path_local

        if lora_count > 0:
            high_lora_node_id = "279"

            for i, lora_pair in enumerate(lora_pairs):
                lora_high = lora_pair.get("high")
                lora_low = lora_pair.get("low")
                lora_name = lora_high or lora_low

                if not lora_name:
                    continue

                if lora_high and lora_low and lora_high != lora_low:
                    logger.info(f"LoRA {i+1}: single-stage workflow uses one branch, applying HIGH variant and ignoring LOW variant.")

                lora_weight = (
                    parse_float(lora_pair.get("high_weight", 1.0), f"lora_pairs[{i}].high_weight")
                    if lora_high
                    else parse_float(lora_pair.get("low_weight", 1.0), f"lora_pairs[{i}].low_weight")
                )
                prompt[high_lora_node_id]["inputs"][f"lora_{i+1}"] = lora_name
                prompt[high_lora_node_id]["inputs"][f"strength_{i+1}"] = lora_weight
                logger.info(f"LoRA {i+1} applied to node 279: {lora_name} with weight {lora_weight}")

        ws_url = f"ws://{server_address}:8188/ws?clientId={request_client_id}"
        logger.info(f"Connecting to WebSocket: {ws_url}")

        http_url = f"http://{server_address}:8188/"
        logger.info(f"Checking HTTP connection to: {http_url}")

        max_http_attempts = 180
        for http_attempt in range(max_http_attempts):
            try:
                urllib.request.urlopen(http_url, timeout=5)
                logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
                break
            except Exception as e:
                logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}")
                if http_attempt == max_http_attempts - 1:
                    raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
                time.sleep(1)

        validate_prompt_node_types(prompt)
        validate_workflow_structure(prompt, workflow_file, required_node_ids)

        ws = websocket.WebSocket()
        ws.settimeout(WS_RECV_TIMEOUT_SECONDS)

        max_attempts = int(180/5)
        for attempt in range(max_attempts):
            try:
                ws.connect(ws_url)
                logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
                break
            except Exception as e:
                logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:
                    raise Exception("웹소켓 연결 시간 초과 (3분)")
                time.sleep(5)

        videos = get_videos(ws, prompt, task_id, request_client_id)

        for node_id in videos:
            if videos[node_id]:
                return videos[node_id][0]

        return {"error": "비디오를 찾을 수 없습니다."}
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        cleanup_temp_dir(temp_dir)

runpod.serverless.start({"handler": handler})
