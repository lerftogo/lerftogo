import cv2
import subprocess
import threading

class BRIOWebcam:
    def __init__(self, card:str,width:int,height:int):
        """
        card: /dev/videoX
        width: width of the image
        height: height of the image
        """
        self.card=card
        self.width=width
        self.height=height
        #try opening the camera
        process = subprocess.Popen(['v4l2-ctl', '-d', card, '-v', 'height={0},width={1},pixelformat={2}'.format(self.height, self.width, 'MJPG')], universal_newlines=True, stdout=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("Unable to start v4l2-cam feed, unknown error")
        self.cap = cv2.VideoCapture(card, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(self.height))
        fourcode=cv2.VideoWriter_fourcc(*'{}'.format('MJPG'))
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcode)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        caps = self._get_capabilities()
        for line in caps:
            print(line)
        print(f"Setting defaults based on the above parameters from v4l2-ctl")
        self.set_defaults()

        #start the threading
        self.frame=None
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def set_value(self, setting, value):
        """
        Set a value for a setting which is available through v4l2-ctl
        setting: the setting to change
        value: the value to change it to
        """
        assert setting in self.settings, f"Setting {setting} not found in available settings"
        subprocess.run(['v4l2-ctl', '-d', self.card, '-c', '{0}={1}'.format(setting, value)], check=True, universal_newlines=True)
    
    def update(self):
        while True:
            if self.cap.isOpened():
                _,self.frame=self.cap.read()

    def get_frame(self):
        """
        Get the latest frame from the camera, if it is available
        """
        frame = self.frame
        self.frame=None
        return frame
    
    @property
    def settings(self):
        """
        Get a list of available settings
        """
        settings_list = []
        capabilites = self._get_capabilities()
        for line in capabilites:
            line = line.strip()
            if "0x" in line and "int" in line and not "flags=inactive" in line:
                setting = line.split('0x', 1)[0].strip()
                settings_list.append(setting)
                    
        for line in capabilites:
            line = line.strip()
            if "0x" in line and not "int" in line:
                setting = line.split('0x', 1)[0].strip()
                settings_list.append(setting)
        return settings_list
    
    def set_defaults(self):
        """
        Sets the default values for all settings given by v4l2-ctl -L
        """
        capabilites = self._get_capabilities()
        for line in capabilites:
            line = line.strip()
            if "0x" in line and "int" in line and not "flags=inactive" in line:
                try:
                    self._split_default_value(line)
                except:
                    pass
                    
        for line in capabilites:
            line = line.strip()
            if "0x" in line and not "int" in line:
                try:
                    self._split_default_value(line)
                except:
                    pass

    def _split_default_value(self, line):
        setting = line.split('0x', 1)[0].strip()
        value = line.split("default=", 1)[1]
        value = int(value.split(' ', 1)[0])
        self.set_value(setting, value)

    def _get_capabilities(self):
        try:
            capread = subprocess.run(['v4l2-ctl', '-d', self.card, '-L'], check=True, universal_newlines=True, stdout=subprocess.PIPE)
        except:
            return
        capabilites = capread.stdout.split('\n')
        return capabilites


if __name__ == "__main__":
    import time
    cam = BRIOWebcam('/dev/video0',1280,720)
    cam.set_value('white_balance_temperature_auto',0)
    cam.set_value('white_balance_temperature',3200)
    cam.set_value('exposure_auto',1)
    cam.set_value('exposure_absolute',31)
    cam.set_value('backlight_compensation',0)
    cam.set_value('gain',255)
    cam.set_value('focus_auto',0)
    cam.set_value('focus_absolute',10)
    while True:
        frame = cam.get_frame()
        if frame is not None:
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            # print('frame avail',frame.shape)
            # time.sleep(.01)