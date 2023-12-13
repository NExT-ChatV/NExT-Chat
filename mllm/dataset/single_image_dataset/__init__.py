from .flickr import FlickrParser, FlickrDataset
from .rec import RECDataset, RECComputeMetrics
from .res import RESDataset, RESComputeMetrics
from .reg import REGDataset, REGCapComputeMetrics, GCDataset
from .caption import CaptionDataset
from .instr import InstructDataset
from .gqa import GQADataset, GQAComputeMetrics
from .clevr import ClevrDataset
from .point_qa import Point_QA_local, Point_QA_twice, V7W_POINT, PointQAComputeMetrics
from .gpt_gen import GPT4Gen
from .vcr import VCRDataset, VCRPredDataset
from .vqav2 import VQAv2Dataset
from .vqaex import VQAEXDataset
from .pope import POPEVQADataset, PopeComputeMetrics
