from typing import List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Boolean, ARRAY, JSON, DateTime, Float, UUID, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class AudioSegment(BaseModel):
    start: Union[str, float]
    end: Union[str, float]
    text: str
    speaker: str

class TranscriptionsResponse(BaseModel):
    segments: List[AudioSegment]

class AgentMetricsAudio(BaseModel):
    agent_opening_within_5_sec: Optional[bool] = None
    agent_used_opening_script: Optional[bool] = None
    agent_used_brand_name: Optional[bool] = None
    agent_used_closing_script: Optional[bool] = None
    agent_used_further_assistance_script: Optional[bool] = None
    agent_used_correct_hold_script: Optional[bool] = None
    agent_excessive_hold_duration: Optional[bool] = None
    agent_explained_issue_to_l2: Optional[bool] = None
    agent_followed_escalation_policy: Optional[bool] = None
    agent_attentive_throughout: Optional[bool] = None
    agent_probing_logical_and_correct: Optional[bool] = None
    agent_allowed_customer_to_complete: Optional[bool] = None
    agent_used_jargon: Optional[bool] = None
    agent_spoke_in_customer_language: Optional[bool] = None
    agent_fumbled_or_stammered: Optional[bool] = None
    agent_shared_proactive_info: Optional[Union[bool, None]] = None
    customer_experience_sentiment: Optional[str] = None  # 'positive', 'negative', or 'neutral'
    agent_pitched_csat: Optional[bool] = None
    agent_verified_customer_details: Optional[bool] = None
    agent_verified_customer_details_reason: Optional[str] = None
    no_of_call_holds: Optional[int] = None
    agent_ensured_warm_transfer: Optional[bool] = None
    agent_ensured_warm_transfer_reason: Optional[str] = None
    agent_used_absurd_or_abusive_language: Optional[bool] = None
    agent_used_absurd_or_abusive_language_reason: Optional[str] = None
    agent_tone_type: Optional[bool] = None
    agent_tone_type_reason: Optional[str] = None
    agent_communicated_tat: Optional[bool] = None
    agent_communicated_tat_reason: Optional[str] = None
    overall_customer_sentiment: Optional[str] = None  # 'positive', 'negative', or 'neutral'
    initial_customer_sentiment: Optional[str] = None  # 'positive', 'negative', or 'neutral'
    closure_customer_sentiment: Optional[str] = None  # 'positive', 'negative', or 'neutral'
    total_hold_duration: Optional[int] = None

    dead_air: Optional[bool] = None
    duration_dead_air: Optional[int] = None
    duration_silence: Optional[int] = None
    call_dropped: Optional[bool] = None
    customer_voc: Optional[str] = None
    closure_timeline: Optional[str] = None

class CallMetricsTranscriptResponse(BaseModel):
    call_category: Optional[List[str]] = None
    sub_call_category: Optional[List[str]] = None
    qrc_category: Optional[List[str]] = None
    field_request_asked: Optional[bool] = None
    error_codes_mentioned: Optional[List[str]] = Field(default=None, alias="Error_codes_mentioned")
    customer_know_the_error_code: Optional[bool] = None
    issue_resolved: Optional[bool] = None
    resolution_turn_around_time: Optional[int] = None
    follow_up_required: Optional[bool] = None
    customer_confirmation_for_resolution: Optional[bool] = None
    offer_pitched: Optional[bool] = None
    no_times_offer_pitched: Optional[int] = None
    customer_reaction_to_offer: Optional[str] = None
    offer_category: Optional[List[str]] = None
    offer_names_pitched: Optional[List[str]] = None
    promotion_of_ott: Optional[bool] = None
    recharge_accepted: Optional[bool] = None
    recharge_delayed_reason: Optional[List[str]] = None
    promise_of_payment: Optional[bool] = None
    channel_mentioned: Optional[List[str]] = None
    name_of_competitor: Optional[List[str]] = None
    d_i_y_pitched: Optional[bool] = None
    pay_later_requested: Optional[bool] = None
    pay_later_provided: Optional[bool] = None
    channel_addition_intended: Optional[bool] = None
    channel_deletion_intended: Optional[bool] = None
    charges_customer_refused: Optional[bool] = None
    charges_customer_refused_reason: Optional[str] = None
    # Additional fields from database table
    no_of_speakers: Optional[int] = None
    first_call_resolution: Optional[bool] = None
    call_category_remark: Optional[str] = None
    channel_category: Optional[List[str]] = None
    price_mentioned: Optional[float] = None
    is_offer_purchased: Optional[bool] = None
    change_in_fmr: Optional[int] = None
    technician_competence: Optional[bool] = None
    customer_satisfaction_over_technician_visit: Optional[str] = None
    technician_behaviour: Optional[str] = None
    customer_technician_complain: Optional[str] = None
    billing_mode: Optional[str] = None
    waivers_requested: Optional[bool] = None
    waivers_offered: Optional[bool] = None
    waiver_type: Optional[str] = None
    promotion_of_products: Optional[bool] = None
    competitor_mentioned: Optional[bool] = None
    qrc_category_remark: Optional[str] = None
    multiple_call_mentioned: Optional[bool] = None
    visit_charges_refused: Optional[bool] = None
    complaining_about_FMR: Optional[bool] = None
    watcho_related_issues: Optional[bool] = None
    app_related_issues: Optional[bool] = None

    offer_purchased: Optional[str] = None
    repeat_call_number: Optional[int] = None
    sub_sub_category: Optional[List[str]] = None


class CallRecord(Base):
    __tablename__ = 'agent_call_record'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    brand = Column(String)
    call_id = Column(String)
    call_start_time = Column(String)
    ucid = Column(String)
    level = Column(String)
    duration = Column(Integer)
    smsid = Column(String)
    vcno = Column(String)
    location = Column(String)
    box_type = Column(String, name='box_type')
    customer_type = Column(String, name='customer_type')
    cas_type = Column(String, name='cas_type')
    call_language = Column(String)
    aon_months = Column(Integer)
    agent_id = Column(String, name='agent_id')
    agent_name = Column(String, name='agent_name')
    centre = Column(String)
    source_of_call = Column(String, name='source_of_call')
    warranty = Column(String)
    customer_segment = Column(String)
    churn_decile = Column(Integer)
    satelite = Column(String)
    status = Column(String)
    watcho_identification = Column(Boolean)
    renewal_due = Column(String)
    last_recharge_date = Column(String)
    current_fmr = Column(Float, name='current_fmr')
    recharge_offer_name = Column(String, name='recharge_offer_name')
    zing_flag = Column(Boolean)
    da_days = Column(Integer)
    customer_segment3 = Column(String, name='payer_category')
    cricket_floaters = Column(Boolean)
    urban_rural = Column(String)
    bsp_user_warranty = Column(Boolean)
    model_name = Column(String)
    pay_later = Column(Boolean)
    waiver_offered = Column(Boolean)
    waiver_availed = Column(Boolean)
    repeat_call = Column(Boolean)
    infant = Column(Boolean)
    troubleshooting_error_codes = Column(Boolean)
    app_user = Column(Boolean)
    fmr = Column(Float)
    smart_user = Column(Boolean)
    watcho_user = Column(Boolean)
    vas_user = Column(Boolean)
    dnd = Column(Boolean)
    acquisation_pack = Column(String)
    acquisation_pack_price = Column(Float)
    current_pack = Column(String)
    current_pack_price = Column(Float)
    agent_opening_within_5_sec = Column(Boolean)
    agent_used_opening_script = Column(Boolean)
    agent_used_brand_name = Column(Boolean)
    agent_used_closing_script = Column(Boolean)
    agent_used_further_assistance_script = Column(Boolean)
    agent_used_correct_hold_script = Column(Boolean)
    agent_excessive_hold_duration = Column(Boolean)
    agent_explained_issue_to_l2 = Column(Boolean)
    agent_followed_escalation_policy = Column(Boolean)
    agent_attentive_throughout = Column(Boolean)
    agent_probing_logical_and_correct = Column(Boolean)
    agent_allowed_customer_to_complete = Column(Boolean)
    agent_used_jargon = Column(Boolean)
    agent_spoke_in_customer_language = Column(Boolean)
    agent_fumbled_or_stammered = Column(Boolean)
    agent_shared_proactive_info = Column(Boolean)
    customer_experience_sentiment = Column(String)
    agent_pitched_csat = Column(Boolean)
    agent_verified_customer_details = Column(Boolean)
    agent_verified_customer_details_reason = Column(Text)
    agent_hold_used_less_than_3_times = Column(Boolean)
    agent_ensured_warm_transfer = Column(Boolean)
    agent_ensured_warm_transfer_reason = Column(Text)
    agent_used_absurd_or_abusive_language = Column(Boolean)
    agent_used_absurd_or_abusive_language_reason = Column(Text)
    agent_tone_type = Column(Boolean)
    agent_tone_type_reason = Column(Text)
    agent_communicated_tat = Column(Boolean)
    agent_communicated_tat_reason = Column(Text)
    overall_customer_sentiment = Column(String)
    initial_customer_sentiment = Column(String)
    closure_customer_sentiment = Column(String)
    dead_air = Column(Boolean)
    duration_dead_air = Column(Integer)
    call_dropped = Column(Boolean)
    customer_voc = Column(Text)
    closure_timeline = Column(Text)
    max_agent_opening_and_closing_score = Column(Integer)
    agent_opening_and_closing_score = Column(Integer)
    max_hold_and_transfer_score = Column(Integer)
    hold_and_transfer_score = Column(Integer)
    max_comprehending_qrc_score = Column(Integer)
    comprehending_qrc_score = Column(Integer)
    max_language_score = Column(Integer)
    language_score = Column(Integer)
    max_proactive_info_score = Column(Integer)
    proactive_info_score = Column(Integer)
    total_score = Column(Integer)
    calculated_score = Column(Integer)
    is_fatal = Column(Boolean)
    no_of_speakers = Column(Integer)
    duration_silence = Column(Integer)
    duration_music = Column(Integer)
    no_of_call_holds = Column(Integer)
    total_hold_duration = Column(Integer)
    issue_resolved = Column(Boolean)
    resolution_turn_around_time = Column(Integer)
    follow_up_required = Column(Boolean)
    first_call_resolution = Column(Boolean)
    repeated_calls = Column(Boolean)
    field_request_asked = Column(Boolean)
    customer_confirmation_for_resolution = Column(Boolean)
    call_escalated = Column(Boolean)
    call_category = Column(ARRAY(String))
    call_category_remark = Column(String)
    sub_call_category = Column(ARRAY(String))
    channel_mentioned = Column(ARRAY(String))
    channel_category = Column(ARRAY(String))
    price_mentioned = Column(Float)
    offer_pitched = Column(Boolean)
    no_times_offer_pitched = Column(Integer)
    customer_reaction_to_offer = Column(String)
    offer_category = Column(ARRAY(String))
    offer_names_pitched = Column(ARRAY(String))
    is_offer_purchased = Column(Boolean)
    offer_purchased = Column(String)
    promotion_of_ott = Column(Boolean)
    recharge_accepted = Column(Boolean)
    recharge_delayed_reason = Column(ARRAY(String))
    change_in_fmr = Column(Integer)
    post_call_recharge_date = Column(DateTime)
    post_call_recharge_days = Column(Integer)
    post_call_recharge_amount = Column(Integer)
    post_call_waiver_credit_amount = Column(Integer)
    technician_competence = Column(Boolean)
    customer_satisfaction_over_technician_visit = Column(String)
    technician_behaviour = Column(String)
    customer_technician_complain = Column(String)
    billing_mode = Column(String)
    waivers_requested = Column(Boolean)
    waivers_offered = Column(Boolean)
    waiver_type = Column(String)
    promotion_of_products = Column(Boolean)
    competitor_mentioned = Column(Boolean)
    name_of_competitor = Column(ARRAY(String))
    qrc_category = Column(ARRAY(String))
    qrc_category_remark = Column(String)
    multiple_call_mentioned = Column(Boolean)
    error_codes_mentioned = Column(ARRAY(String))
    customer_know_the_error_code = Column(Boolean)
    promise_of_payment = Column(Boolean)
    repeat_call_number = Column(Integer)
    free_of_cost = Column(Boolean)
    pay_later_requested = Column(Boolean)
    d_i_y_pitched = Column(Boolean)
    visit_charges_refused = Column(Boolean)
    pay_later_provided = Column(Boolean)
    channel_addition_intended = Column(Boolean)
    channel_deletion_intended = Column(Boolean)
    charges_customer_refused = Column(Boolean)
    charges_customer_refused_reason = Column(String)
    complaining_about_FMR = Column(Boolean)
    l2_transfer = Column(Boolean)
    watcho_related_issues = Column(Boolean)
    app_related_issues = Column(Boolean)
    transcriptions = Column(JSON)
    skill_name = Column(String)
    lob = Column(String)
    aon_days = Column(Float)
    sub_sub_category = Column(ARRAY(String))

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class BatchJob(Base):
    """Database model for tracking batch jobs."""
    __tablename__ = 'batch_jobs'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_uuid = Column(String, unique=True, nullable=False, index=True)
    
    # Generic batch job IDs (formerly Gemini-specific)
    transcription_batch_id = Column(String, nullable=True)
    agent_metrics_batch_id = Column(String, nullable=True)
    
    transcription_input_path = Column(String, nullable=True)
    agent_metrics_input_path = Column(String, nullable=True)
    transcription_output_path = Column(String, nullable=True)
    agent_metrics_output_path = Column(String, nullable=True)
    
    status = Column(String, default='processing', nullable=False)
    error_message = Column(Text, nullable=True)
    
    total_messages = Column(Integer, default=0)
    total_divisions = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    year = Column(String, nullable=True)
    month = Column(String, nullable=True)
    day = Column(String, nullable=True)
    messages = Column(JSON, nullable=True)
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
