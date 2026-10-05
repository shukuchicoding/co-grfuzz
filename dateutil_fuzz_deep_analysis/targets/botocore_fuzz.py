import io
import botocore
from botocore import serialize
from botocore.model import ServiceModel

# fuzz_serialization.py

def fuzz(buf):
    if len(buf) < 1:
        return
    
    # Tạo một protocol từ byte đầu tiên
    protocol_index = buf[0] % 3
    protocols = ['rest-xml', 'json', 'query']
    protocol_str = protocols[protocol_index]

    model = {
        'metadata': {'protocol': protocol_str, 'apiVersion': '2022-01-01'},
        'documentation': '',
        'operations': {
            'FuzzOperation': {
                'name': 'FuzzOperation',
                'http': {
                    'method': 'POST',
                    'requestUri': '/',
                },
                'input': {'shape': 'FuzzInputShape'},
            }
        },
        'shapes': {
            'FuzzInputShape': {
                'type': 'structure',
                'members': {
                    'Blob': {'shape': 'BlobType'},
                },
            },
            'BlobType': {
                'type': 'blob',
            },
        },
    }

    service_model = ServiceModel(model)
    request_serializer = serialize.create_serializer(service_model.metadata['protocol'])

    try:
        request_serializer.serialize_to_request(
            io.BytesIO(buf),
            service_model.operation_model('FuzzOperation')
        )
    except botocore.exceptions.ParamValidationError:
        pass
    except Exception:
        pass