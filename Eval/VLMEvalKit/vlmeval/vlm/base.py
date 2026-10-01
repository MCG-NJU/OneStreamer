class BaseModel:
    def __init__(self, **kwargs): pass
    def generate(self,message,dataset=None):
        if isinstance(message,str): message=[dict(type='text',value=message)]
        return self.generate_inner(message,dataset=dataset)
    def use_custom_prompt(self,dataset): return False
