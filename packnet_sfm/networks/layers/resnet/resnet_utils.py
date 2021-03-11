from .resnet import *


__all__ = ['load_model_url', 'load_resnet_model', 'load_resnet_block']


def load_model_url(enc_type):
    model_urls = {'resnet18':              'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                  'resnet34':              'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                  'resnet50':              'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                  'resnet101':             'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                  'resnet152':             'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                  'resnext50_32x4d':       'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
                  'resnext101_32x8d':      'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
                  'wide_resnet50_2':       'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
                  'wide_resnet101_2':      'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
                  'resnet18_ssl':          'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth',
                  'resnet50_ssl':          'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth',    
                  'resnext50_32x4d_ssl':   'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth',
                  'resnext101_32x4d_ssl':  'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth',
                  'resnext101_32x8d_ssl':  'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth',
                  'resnext101_32x16d_ssl': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth',
                  'resnet18_swsl':         'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth',
                  'resnet50_swsl':         'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth',    
                  'resnext50_32x4d_swsl': ' https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth',
                  'resnext101_32x4d_swsl':' https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
                  'resnext101_32x8d_swsl':' https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
                  'resnext101_32x16d_swsl':'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth'
                }
    
    if enc_type not in model_urls:
            raise ValueError("{} is not valid network".format(enc_type))
            
    return model_urls[enc_type]

def load_resnet_model(enc_type):
      nets = {"resnet18": resnet18,                             
              "resnet34": resnet34,   
              "resnet50": resnet50, 
              "resnet101": resnet101,                      
              "resnet152": resnet152, 
              "resnext50_32x4d": resnext50_32x4d, 
              "resnext101_32x8d": resnext101_32x8d,
              "wide_resnet50_2": wide_resnet50_2,            
              "wide_resnet101_2": wide_resnet101_2, 
              "resnet18_ssl": resnet18_ssl,                  
              "resnet50_ssl": resnet50_ssl,
              "resnext50_32x4d_ssl": resnext50_32x4d_ssl,     
              "resnext101_32x4d_ssl": resnext101_32x4d_ssl, 
              "resnext101_32x8d_ssl": resnext101_32x8d_ssl,   
              "resnext101_32x16d_ssl": resnext101_32x16d_ssl,
              "resnet18_swsl": resnet18_swsl,                 
              "resnet50_swsl": resnet50_swsl, 
              "resnext50_32x4d_swsl": resnext50_32x4d_swsl,   
              "resnext101_32x4d_swsl": resnext101_32x4d_swsl,
              "resnext101_32x8d_swsl": resnext101_32x8d_swsl, 
              "resnext101_32x16d_swsl": resnext101_32x16d_swsl}
  
      if enc_type not in nets:
            raise ValueError("{} is not valid network".format(enc_type))
            
      resnet_model = nets[enc_type]
      return resnet_model
  
def load_resnet_block(enc_type):
      blocks = {"resnet18": [2, 2, 2, 2],                             
                "resnet34": [3, 4, 6, 3],   
                "resnet50": [3, 4, 6, 3], 
                "resnet101": [3, 4, 23, 3],                      
                "resnet152": [3, 8, 36, 3], 
                "resnext50_32x4d": [3, 4, 6, 3], 
                "resnext101_32x8d": [3, 4, 23, 3],
                "wide_resnet50_2": [3, 4, 6, 3],            
                "wide_resnet101_2": [3, 4, 23, 3], 
                "resnet18_ssl": [2, 2, 2, 2],                  
                "resnet50_ssl": [3, 4, 6, 3],
                "resnext50_32x4d_ssl": [3, 4, 6, 3],     
                "resnext101_32x4d_ssl": [3, 4, 23, 3], 
                "resnext101_32x8d_ssl": [3, 4, 23, 3],   
                "resnext101_32x16d_ssl": [3, 4, 23, 3],
                "resnet18_swsl": [2, 2, 2, 2],                 
                "resnet50_swsl": [3, 4, 6, 3], 
                "resnext50_32x4d_swsl": [3, 4, 6, 3],   
                "resnext101_32x4d_swsl": [3, 4, 23, 3],
                "resnext101_32x8d_swsl": [3, 4, 23, 3], 
                "resnext101_32x16d_swsl": [3, 4, 23, 3]}
      
      if enc_type not in blocks:
            raise ValueError("{} is not valid network".format(enc_type))
      
      resnet_block = blocks[enc_type]
      return resnet_block
