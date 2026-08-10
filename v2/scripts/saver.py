import json
import inspect

def saveable(cls):
    original_init = cls.__init__

    def serialize_value(value):
        if hasattr(value, '__get_config__'):
            return {**value.__get_config__(), "_type": "saveable"}
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [serialize_value(v) for v in value]
        else:
            try:
                json.dumps(value)
                return value
            except (TypeError, OverflowError):
                return str(value)

    def new_init(self, *args, **kwargs):
        sig = inspect.signature(original_init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        self.init_args = {}
        for key, value in bound.arguments.items():
            if key != 'self':
                self.init_args[key] = serialize_value(value)
        original_init(self, *args, **kwargs)

    def __get_config__(self):
        return {
            'class_name': cls.__name__,
            'module': cls.__module__,
            'init_args': self.init_args
        }

    def save_config(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.__get_config__(), f, indent=2)

    cls.__init__ = new_init
    cls.__get_config__ = __get_config__
    cls.save_config = save_config
    return cls

def load_from_config_content(config_content):
    class_name = config_content['class_name']
    module_name = config_content.get('module')
    init_args = config_content.get('init_args', {})
    
    def restore_value(value):
        if isinstance(value, dict):
            if '_type' in value and value['_type'] == 'saveable':
                return load_from_config_content(value)
            return {k: restore_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore_value(item) for item in value]
        return value
    
    init_args = {k: restore_value(v) for k, v in init_args.items()}
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    if model_class is None:
        raise ValueError(f"Класс {class_name} из модуля {module} не найден")

    return model_class(**init_args)

def load_from_config(filepath):
    with open(filepath, 'r') as f:
        return load_from_config_content(json.load(f))
