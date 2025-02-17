# A1111 Stable Diffusion | RunPod Serverless Worker

This is the source code for a [RunPod](https://runpod.io?ref=2xxro4sy)
Serverless worker that uses the [Automatic1111 Stable Diffusion API](
https://github.com/AUTOMATIC1111/stable-diffusion-webui) for inference.

## Model

The model(s) for inference will be loaded from a RunPod
Network Volume.

## Testing

1. [Local Testing](docs/testing/local.md)
2. [RunPod Testing](docs/testing/runpod.md)

## Installing, Building and Deploying the Serverless Worker

1. [Install Automatic1111 Web UI on your Network Volume](
docs/installing.md)
2. [Building the Docker image](docs/building.md)
3. [Deploying on RunPod Serveless](docs/deploying.md)
4. [Frequently Asked Questions](docs/faq.md)

## RunPod API Endpoint

You can send requests to your RunPod API Endpoint using the `/run`
or `/runsync` endpoints.

Requests sent to the `/run` endpoint will be handled asynchronously,
and are non-blocking operations.  Your first response status will always
be `IN_QUEUE`.  You need to send subsequent requests to the `/status`
endpoint to get further status updates, and eventually the `COMPLETED`
status will be returned if your request is successful.

Requests sent to the `/runsync` endpoint will be handled synchronously
and are blocking operations.  If they are processed by a worker within
90 seconds, the result will be returned in the response, but if
the processing time exceeds 90 seconds, you will need to handle the
response and request status updates from the `/status` endpoint until
you receive the `COMPLETED` status which indicates that your request
was successful.

### RunPod API Examples

#### A1111 APIs

* [Get ControlNet Models](docs/api/a1111/get-controlnet-models.md)
* [Get Embeddings](docs/api/a1111/get-embeddings.md)
* [Get Face Restorers](docs/api/a1111/get-face-restorers.md)
* [Get Hypernetworks](docs/api/a1111/get-hypernetworks.md)
* [Get Loras](docs/api/a1111/get-loras.md)
* [Get Memory](docs/api/a1111/get-memory.md)
* [Get Models](docs/api/a1111/get-models.md)
* [Get Options](docs/api/a1111/get-options.md)
* [Get Prompt Styles](docs/api/a1111/get-prompt-styles.md)
* [Get Real-ESRGAN Models](docs/api/a1111/get-realesrgan-models.md)
* [Get Samplers](docs/api/a1111/get-samplers.md)
* [Get Script Info](docs/api/a1111/get-script-info.md)
* [Get Scripts](docs/api/a1111/get-scripts.md)
* [Get Upscalers](docs/api/a1111/get-upscalers.md)
* [Get VAE](docs/api/a1111/get-vae.md)
* [Image to Image](docs/api/a1111/img2img.md)
* [Image to Image with ControlNet](docs/api/a1111/img2img-controlnet.md)
* [Interrogate](docs/api/a1111/interrogate.md)
* [Refresh Checkpoints](docs/api/a1111/refresh-checkpoints.md)
* [Refresh Loras](docs/api/a1111/refresh-loras.md)
* [Set Model](docs/api/a1111/set-model.md)
* [Set VAE](docs/api/a1111/set-vae.md)
* [Text to Image](docs/api/a1111/txt2img.md)
* [Text to Image with ReActor](docs/api/a1111/txt2img-reactor.md)
* [Text to Image with ADetailer](docs/api/a1111/txt2img-adetailer.md)

#### Helper APIs

* [File Download](docs/api/helper/download.md)
* [Huggingface Sync](docs/api/helper/sync.md)

### Optional Webhook Callbacks

You can optionally [Enable a Webhook](docs/api/helper/webhook.md).

### Endpoint Status Codes

| Status      | Description                                                                                                                     |
|-------------|---------------------------------------------------------------------------------------------------------------------------------|
| IN_QUEUE    | Request is in the queue waiting to be picked up by a worker.  You can call the `/status` endpoint to check for status updates.  |
| IN_PROGRESS | Request is currently being processed by a worker.  You can call the `/status` endpoint to check for status updates.             |
| FAILED      | The request failed, most likely due to encountering an error.                                                                   |
| CANCELLED   | The request was cancelled.  This usually happens when you call the `/cancel` endpoint to cancel the request.                    |
| TIMED_OUT   | The request timed out.  This usually happens when your handler throws some kind of exception that does return a valid response. |
| COMPLETED   | The request completed successfully and the output is available in the `output` field of the response.                           |

## Serverless Handler

The serverless handler (`rp_handler.py`) is a Python script that handles
the API requests to your Endpoint using the [runpod](https://github.com/runpod/runpod-python)
Python library.  It defines a function `handler(event)` that takes an
API request (event), runs the inference using the model(s) from your
Network Volume with the `input`, and returns the `output`
in the JSON response.

## Acknowledgements

- [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Generative Labs YouTube Tutorials](https://www.youtube.com/@generativelabs)

## Additional Resources

- [Postman Collection for this Worker](RunPod_A1111_Worker.postman_collection.json)
- [Generative Labs YouTube Tutorials](https://www.youtube.com/@generativelabs)
- [Getting Started With RunPod Serverless](https://trapdoor.cloud/getting-started-with-runpod-serverless/)
- [Serverless | Create a Custom Basic API](https://blog.runpod.io/serverless-create-a-basic-api/)

## Community and Contributing

Pull requests and issues on [GitHub](https://github.com/ashleykleynhans/runpod-worker-a1111)
are welcome. Bug fixes and new features are encouraged.

You can contact me and get help with deploying your Serverless
worker to RunPod on the RunPod Discord Server below,
my username is **ashleyk**.

<a target="_blank" href="https://discord.gg/pJ3P2DbUUq">![Discord Banner 2](https://discordapp.com/api/guilds/912829806415085598/widget.png?style=banner2)</a>

## Appreciate my work?

<a href="https://www.buymeacoffee.com/ashleyk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
#   r u n p o d - a u t o 1 1 1 1  
 #   r u n p o d - a u t o 1 1 1 1  
 