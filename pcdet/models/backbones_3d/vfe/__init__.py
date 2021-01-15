from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .vfe_template import VFETemplate
# from .dynamic_pillar_vfe import DynamicPillarVFE
from .pillar_vfe_2view import MultiViewPillarNet 
from .attention_2view_pillar_vfe import Attention2ViewPillarNet
from .view_wise_attention_vfe import ViewWiseAttentionPillarNet
from .multiview_conv import Multiview2Conv
from .dynamic_pillar_vfe_multi_view import MultiViewDynamicPillarVFE
__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    # 'DynamicPillarVFE': DynamicPillarVFE,
    'MultiViewPillarNet': MultiViewPillarNet,
    'Attention2ViewPillarNet': Attention2ViewPillarNet,
    'ViewWiseAttentionPillarNet': ViewWiseAttentionPillarNet,
    'Multiview2Conv': Multiview2Conv,
    'MultiViewDynamicPillarVFE': MultiViewDynamicPillarVFE

}
