from dataclasses import dataclass
from typing import Literal, Optional
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.material.elastic_material import LinearElasticMaterial

from .base import MaterialInterpolation

@dataclass
class ElasticMaterialConfig:
    """Configuration class for elastic material properties."""
    
    elastic_modulus: float = 1.0
    minimal_modulus: float = 1e-9
    poisson_ratio: float = 0.3
    plane_assumption: Literal["plane_stress", "plane_strain", "3d"] = "plane_stress"

class ElasticMaterialProperties(LinearElasticMaterial):
    """Class for elastic material properties with interpolation capabilities."""

    def __init__(self, 
                config: ElasticMaterialConfig, 
                rho: TensorLike,
                interpolation_model: MaterialInterpolation
                ):
        """
        Initialize elastic material properties.
        
        Args:
            config : Material configuration
            rho : Density field
            interpolation_model : Material interpolation model
        """
        if not isinstance(config, ElasticMaterialConfig):
            raise TypeError("'config' must be an instance of ElasticMaterialConfig")
        
        if rho is None or not isinstance(rho, TensorLike):
            raise TypeError("'rho' must be of type TensorLike and cannot be None")
        
        if interpolation_model is None or not isinstance(interpolation_model, MaterialInterpolation):
            raise TypeError("'interpolation_model' must be an instance of MaterialInterpolation")
        
        super().__init__(
                        name="ElasticMaterialProperties",
                        elastic_modulus=config.elastic_modulus,
                        poisson_ratio=config.poisson_ratio,
                        hypo=config.plane_assumption
                    )
        
        # 作为普通实例变量
        self.config = config
        self.interpolation_model = interpolation_model
        
        # 密度场需要动态更新，保持为属性
        self._rho = rho
        
        # 初始化基础材料
        self._base_material = LinearElasticMaterial(
                                    name="BaseMaterial",
                                    elastic_modulus=1.0,
                                    poisson_ratio=config.poisson_ratio,
                                    hypo=config.plane_assumption
                                )
        
    @property
    def rho(self) -> TensorLike:
        """Get the density field."""
        return self._rho
    
    @rho.setter
    def rho(self, value: TensorLike):
        """Set the density field."""
        if value is None or not isinstance(value, TensorLike):
            raise TypeError("'rho' must be of type TensorLike and cannot be None")
        self._rho = value


    @property
    def base_elastic_material(self) -> LinearElasticMaterial:
        """获取 E=1 时的基础材料属性"""
        return self._base_material

    def material_model(self) -> TensorLike:
        """Calculate interpolated Young's modulus."""
        E = self.interpolation_model.calculate_property(
                                        self.rho,
                                        self.config.elastic_modulus,
                                        self.config.minimal_modulus,
                                        self.interpolation_model.penalty_factor
                                    )
        return E

    def material_model_derivative(self) -> TensorLike:
        """Calculate derivative of interpolated Young's modulus."""
        dE = self.interpolation_model.calculate_property_derivative(
                                            self.rho,
                                            self.config.elastic_modulus,
                                            self.config.minimal_modulus,
                                            self.interpolation_model.penalty_factor
                                        )
        return dE

    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """
        Calculate the elastic matrix D for each element.
        
        Args:
            bcs (Optional[TensorLike]): Boundary conditions
            
        Returns:
            TensorLike: Elastic matrix D for each element
        """
        if self.rho is None:
            raise ValueError("Density rho must be set for MaterialProperties.")
        
        E = self.material_model()
        base_D = super().elastic_matrix()
        D = E[:, None, None, None] * base_D

        return D