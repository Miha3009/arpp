import json
import inspect

def saveable(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        sig = inspect.signature(original_init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        self.init_args = {}
        for key, value in bound.arguments.items():
            if key != 'self':
                self.init_args[key] = value        
        original_init(self, *args, **kwargs)

    def save_config(self, filepath):
        config = {
            'class_name': cls.__name__,
            'module': cls.__module__,
            'init_args': self.init_args
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    cls.__init__ = new_init
    cls.save_config = save_config    
    return cls

def load_from_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    class_name = config['class_name']
    module_name = config.get('module')
    init_args = config.get('init_args', {})
    
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    if model_class is None:
        raise ValueError(f"Класс {class_name} из модуля {module} не найден")

    return model_class(**init_args)
