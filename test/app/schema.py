#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    pelvic_incidence: float = Field(..., example=47.903565,  title=' pelvic_incidence')
    pelvic_tilt: float = Field(..., example=13.616688,  title='spelvic_tilt ')
    lumbar_lordosis_angle: float = Field(..., example=36.000000,  title='lumbar_lordosis_angle')
    sacral_slope: float = Field(..., example=34.286877,  title='sacral_slope')
        
    pelvic_radius: float = Field(..., example=117.449062,  title='pelvic_radius')
    degree_spondylolisthesis: float = Field(..., example=-4.245395,  title='degree_spondylolisthesis')
    pelvic_slope: float = Field(..., example=0.129744,  title='pelvic_slope')
    direct_tilt: float = Field(..., example=7.843300,  title='direct_tilt')
    
    thoracic_slope: float = Field(..., example=14.748400,  title='thoracic_slope')
    cervical_tilt: float = Field(..., example=8.517070,  title='cervical_tilt')
    sacrum_angle: float = Field(..., example=-15.728927,  title='sacrum_angle')
    scoliosis_slope: float = Field(..., example=11.547200,  title='scoliosis_slope')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    logits: float = Field(..., example=0, title='logits')
    # versicolor: float = Field(..., example=0.000015, title='Probablity for class versicolor')
    # virginica: float = Field(..., example=0.012459, title='Probablity for class virginica')
    # pred: str = Field(..., example='versicolor', title='Predicted class with highest probablity')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')