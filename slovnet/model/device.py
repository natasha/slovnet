

class DeviceMixin:
    @property
    def device(self):
        for parameter in self.parameters():
            return parameter.device
