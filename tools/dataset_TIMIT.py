from torch.utils.data import Dataset
import glob
import soundfile as sf


class DatasetTIMIT(Dataset):
    def __init__(self,
                 mixture_path,
                 clean_path,
                 is_check):
        super(DatasetTIMIT, self).__init__()

        self.mixture_paths = self.get_path(mixture_path)
        self.clean_paths = self.get_path(clean_path)
        if is_check:
            self.check_data(self.mixture_paths, self.clean_paths)
        self.matched_mixture_paths, self.matched_clean_paths = self.match_data(self.mixture_paths, self.clean_paths)
        self.length = len(self.matched_clean_paths)

    def __getitem__(self, index: int):
        mixture_path = self.matched_mixture_paths[index]
        clean_path = self.matched_clean_paths[index]
        mixture, sr = sf.read(mixture_path, dtype="float32")
        clean, sr = sf.read(clean_path, dtype="float32")
        assert sr == 16000
        assert mixture.shape == clean.shape
        assert mixture.shape == clean.shape
        return mixture, clean, mixture.shape[0], mixture_path.split("/")[-1]

    def __len__(self):
        return self.length

    @staticmethod
    def get_path(root):
        """
        :param root:
        :return:
        """
        path_list = []
        for name in glob.glob(root + "/*"):
            if name.split(".")[-1] in ['wav', 'WAV']:
                path_list.append(name)
        return path_list

    @staticmethod
    def check_data(mixture_paths, clean_paths):
        """
        :param mixture_paths:
        :param clean_paths:
        :return:
        """
        for mixture_path in mixture_paths:
            audio_name = mixture_path.split("/")[-1].split("+")[-1].strip('-20_')
            is_ok = 0
            for clean_path in clean_paths:
                if audio_name in clean_path:
                    is_ok = 1
            if is_ok == 0:
                raise ValueError(f"mixture audio: {audio_name} not in clean path")

    @staticmethod
    def match_data(mixture_paths, clean_paths):
        matched_mixture_paths = []
        matched_clean_paths = []
        for mixture_path in mixture_paths:
            audio_name = mixture_path.split("/")[-1].split("+")[-1].strip('-20_')
            for clean_path in clean_paths:
                if audio_name in clean_path:
                    matched_mixture_paths.append(mixture_path)
                    matched_clean_paths.append(clean_path)
        return matched_mixture_paths, matched_clean_paths
