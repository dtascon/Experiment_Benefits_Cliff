from otree.api import *
import csv
import random
import string
from pathlib import Path

doc = """
Enhanced PRD / Benefit Cliff Experiment - 20 Strategic Vignettes
4-Stage Structure: Stage 1 (Text) → Stage 2 (Dashboard/ChatGPT) → Feedback → Reconsideration
"""

# ----------------------------
# Configuration
# ----------------------------

SEED = 123
N_VIGNETTES = 20

WEEKS_PER_MONTH = 52 / 12
PAYROLL_TAX_RATE = 0.0765
INCOME_TAX_RATE = 0.10


class LocationBenefits:
    """Location and family-specific benefit rules based on PRD Dashboard"""

    def __init__(self, state="PA", county="Allegheny", num_children=2, child_ages=None, has_disability=False):
        self.state = state
        self.county = county
        self.num_children = num_children
        self.child_ages = child_ages or [3, 6]
        self.has_disability = has_disability
        self.set_rules()

    def set_rules(self):
        """Set state/county specific parameters"""

        if self.state == "PA":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 1806.0
            self.medicaid_child_limit = 4168.0
            self.ccdf_max = 800.0
            self.ccdf_copay_threshold = 1800.0
            self.ccdf_exit = 3200.0
            self.tanf_max = 403.0
            self.tanf_limit = 614.0

        elif self.state == "GA":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 503.0
            self.medicaid_child_limit = 4168.0
            self.ccdf_max = 750.0
            self.ccdf_copay_threshold = 1600.0
            self.ccdf_exit = 3000.0
            self.tanf_max = 280.0
            self.tanf_limit = 500.0

        elif self.state == "TX":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 288.0
            self.medicaid_child_limit = 3629.0
            self.ccdf_max = 700.0
            self.ccdf_copay_threshold = 1500.0
            self.ccdf_exit = 2800.0
            self.tanf_max = 303.0
            self.tanf_limit = 450.0

        self.eitc_phase_in = 0.40 if self.num_children >= 3 else (0.34 if self.num_children == 2 else 0.15)
        self.eitc_max = 6935.0 if self.num_children >= 3 else (5920.0 if self.num_children == 2 else 3584.0)
        self.eitc_plateau_start = 15950.0 / 12
        self.eitc_phase_out_start = 25500.0 / 12
        self.eitc_phase_out_rate = 0.2106
        self.eitc_phase_out_end = 57000.0 / 12

        if self.has_disability:
            self.ssdi_benefit = 1537.0
            self.ssi_benefit = 943.0

    def snap(self, monthly_gross: float) -> float:
        if monthly_gross > self.snap_max_gross:
            return 0.0
        net_income = monthly_gross - 200
        if net_income > self.snap_net_limit:
            return 0.0
        benefit = self.snap_max_benefit - (0.30 * max(0, net_income))
        return max(0.0, round(benefit, 2))

    def medicaid(self, monthly_gross: float) -> float:
        if self.num_children > 0:
            if monthly_gross <= self.medicaid_adult_limit:
                return self.medicaid_value * (1 + self.num_children * 0.3)
            elif monthly_gross <= self.medicaid_child_limit:
                return self.medicaid_value * (self.num_children * 0.3)
        else:
            if monthly_gross <= self.medicaid_adult_limit:
                return self.medicaid_value
        return 0.0

    def ccdf(self, monthly_gross: float) -> float:
        if self.num_children == 0 or all(age >= 13 for age in self.child_ages):
            return 0.0
        if monthly_gross <= self.ccdf_copay_threshold:
            return self.ccdf_max
        elif monthly_gross <= self.ccdf_exit:
            phase_range = self.ccdf_exit - self.ccdf_copay_threshold
            phase_amount = ((monthly_gross - self.ccdf_copay_threshold) / phase_range) * self.ccdf_max
            return max(0.0, round(self.ccdf_max - phase_amount, 2))
        return 0.0

    def tanf(self, monthly_gross: float) -> float:
        if self.num_children == 0:
            return 0.0
        if monthly_gross > self.tanf_limit:
            return 0.0
        benefit = self.tanf_max - (0.50 * monthly_gross)
        return max(0.0, round(benefit, 2))

    def wic(self, monthly_gross: float) -> float:
        if self.num_children == 0 or all(age >= 5 for age in self.child_ages):
            return 0.0
        wic_limit = 2177.0
        if monthly_gross <= wic_limit:
            return 50.0
        return 0.0

    def head_start(self, monthly_gross: float) -> float:
        has_preschooler = any(3 <= age <= 5 for age in self.child_ages)
        if not has_preschooler:
            return 0.0
        head_start_limit = 2177.0
        if monthly_gross <= head_start_limit:
            return 300.0
        return 0.0

    def eitc(self, annual_gross: float) -> float:
        if self.num_children == 0:
            return 0.0
        monthly = annual_gross / 12
        if monthly < self.eitc_plateau_start:
            credit = annual_gross * self.eitc_phase_in
        elif monthly < self.eitc_phase_out_start:
            credit = self.eitc_max
        elif monthly < self.eitc_phase_out_end:
            excess = annual_gross - (self.eitc_phase_out_start * 12)
            credit = self.eitc_max - (excess * self.eitc_phase_out_rate)
        else:
            credit = 0.0
        return max(0.0, round(credit / 12, 2))

    def disability_benefits(self, monthly_gross: float) -> dict:
        if not self.has_disability:
            return {"ssdi": 0.0, "ssi": 0.0}
        ssdi = self.ssdi_benefit
        ssi_limit = 943.0
        if monthly_gross < 1000:
            ssi = max(0.0, ssi_limit - (0.50 * monthly_gross))
        else:
            ssi = 0.0
        return {"ssdi": ssdi, "ssi": ssi}

    def total_benefits(self, monthly_gross: float) -> dict:
        annual_gross = monthly_gross * 12
        snap = self.snap(monthly_gross)
        med = self.medicaid(monthly_gross)
        ccdf = self.ccdf(monthly_gross)
        tanf = self.tanf(monthly_gross)
        wic = self.wic(monthly_gross)
        head_start = self.head_start(monthly_gross)
        eitc = self.eitc(annual_gross)
        disability = self.disability_benefits(monthly_gross)
        total = snap + med + ccdf + tanf + wic + head_start + eitc + disability["ssdi"] + disability["ssi"]
        return dict(
            snap=snap, medicaid=med, ccdf=ccdf, tanf=tanf,
            wic=wic, head_start=head_start, eitc=eitc,
            ssdi=disability["ssdi"], ssi=disability["ssi"],
            benefits_total=total,
        )


def monthly_gross(wage: float, hours_per_week: float) -> float:
    return wage * hours_per_week * WEEKS_PER_MONTH


def monthly_net_earnings(gross: float) -> float:
    after_payroll = gross * (1.0 - PAYROLL_TAX_RATE)
    after_income_tax = after_payroll * (1.0 - INCOME_TAX_RATE)
    return after_income_tax


def compute_bundle(wage: float, hours: float, location: LocationBenefits) -> dict:
    gross = monthly_gross(wage, hours)
    net_earn = monthly_net_earnings(gross)
    ben = location.total_benefits(gross)
    true_net_income = net_earn + ben["benefits_total"]
    return dict(
        monthly_gross=round(gross, 2),
        monthly_net_earnings=round(net_earn, 2),
        snap=round(ben["snap"], 2), medicaid=round(ben["medicaid"], 2),
        ccdf=round(ben["ccdf"], 2), tanf=round(ben["tanf"], 2),
        wic=round(ben["wic"], 2), head_start=round(ben["head_start"], 2),
        eitc=round(ben["eitc"], 2), ssdi=round(ben["ssdi"], 2),
        ssi=round(ben["ssi"], 2), benefits_total=round(ben["benefits_total"], 2),
        true_net_income=round(true_net_income, 2),
    )


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def vignette_csv_path() -> Path:
    return app_dir() / "vignettes_20_strategic.csv"


def generate_vignettes_csv(n: int, seed: int) -> None:
    """Generate 20 vignettes: 3 geometries × 3 opportunities × 2 reps + 2 attention checks"""
    rng = random.Random(seed)

    locations = [
        ("PA", "Allegheny"), ("GA", "Fulton"), ("TX", "Harris")
    ]

    family_types = [
        {"num_children": 2, "child_ages": [3, 6], "has_disability": False, "adult_age": 32, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single parent (age 32), 2 children (ages 3, 6)"},
        {"num_children": 1, "child_ages": [4], "has_disability": False, "adult_age": 29, "is_married": False,
         "spouse_age": None, "other_adults": 1,
         "desc": "Single parent (age 29), 1 child (age 4), living with 1 other adult"},
        {"num_children": 2, "child_ages": [5, 9], "has_disability": False, "adult_age": 36, "is_married": True,
         "spouse_age": 38, "other_adults": 0, "desc": "Married parents (ages 36 & 38), 2 children (ages 5, 9)"},
        {"num_children": 0, "child_ages": [], "has_disability": False, "adult_age": 28, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single adult (age 28), no children"},
    ]

    opportunity_descriptions = {
        "low": {
            "current_job": "Retail cashier at local store",
            "offer_job": "Retail cashier at different store (same responsibilities)",
            "mobility_text": "Same job type, no promotion path mentioned"
        },
        "medium": {
            "current_job": "Part-time retail associate",
            "offer_job": "Full-time administrative assistant at local office",
            "mobility_text": "Office environment, some skill development, modest promotion potential"
        },
        "high": {
            "current_job": "Part-time service worker",
            "offer_job": "Full-time entry-level position at established company",
            "mobility_text": "Professional experience, explicit promotion pathway to supervisor ($35k+), employer health benefits"
        }
    }

    cliff_scenarios = [
        {"current_wage": 11, "offer_wage": 14, "hours": 30},
        {"current_wage": 12, "offer_wage": 15, "hours": 28},
    ]

    plateau_scenarios = [
        {"current_wage": 14, "offer_wage": 15, "hours": 32},
        {"current_wage": 15, "offer_wage": 16, "hours": 30},
    ]

    positive_scenarios = [
        {"current_wage": 10, "offer_wage": 12, "hours": 22},
        {"current_wage": 15, "offer_wage": 20, "hours": 35},
    ]

    rows = []
    vignette_counter = 1

    # Generate 2 attention checks
    for i in range(2):
        family = rng.choice(family_types)
        state, county = rng.choice(locations)

        location = LocationBenefits(
            state=state, county=county,
            num_children=family["num_children"],
            child_ages=family["child_ages"],
            has_disability=family["has_disability"]
        )

        current_wage = 10.0
        offer_wage = 18.0
        hours = 25

        cur = compute_bundle(current_wage, hours, location)
        off = compute_bundle(offer_wage, hours, location)

        household_size = 1 + (1 if family["is_married"] else 0) + family["other_adults"] + family["num_children"]
        child_ages_str = ", ".join(str(age) for age in family["child_ages"]) if family["num_children"] > 0 else "none"

        row = dict(
            vignette_id=vignette_counter,
            scenario_type="attention_check",
            target_type="attention_check",
            opportunity_level="high",
            current_job_desc=opportunity_descriptions["high"]["current_job"],
            offer_job_desc=opportunity_descriptions["high"]["offer_job"],
            mobility_description=opportunity_descriptions["high"]["mobility_text"],
            state=state, county=county,
            family_description=family["desc"],
            num_children=family["num_children"],
            has_disability=int(family["has_disability"]),
            adult_age=family["adult_age"],
            is_married=int(family["is_married"]),
            spouse_age=family["spouse_age"] if family["spouse_age"] is not None else 0,
            other_adults=family["other_adults"],
            household_size=household_size,
            child_ages_str=child_ages_str,
            current_hourly_wage=round(current_wage, 2),
            current_hours=int(hours),
            offer_hourly_wage=round(offer_wage, 2),
            offer_hours=int(hours),
            cur_true_net_income=cur["true_net_income"],
            off_true_net_income=off["true_net_income"],
            cur_monthly_gross=cur["monthly_gross"],
            off_monthly_gross=off["monthly_gross"],
            cur_monthly_net_earnings=cur["monthly_net_earnings"],
            off_monthly_net_earnings=off["monthly_net_earnings"],
            cur_snap=cur["snap"], off_snap=off["snap"],
            cur_medicaid=cur["medicaid"], off_medicaid=off["medicaid"],
            cur_ccdf=cur["ccdf"], off_ccdf=off["ccdf"],
            cur_tanf=cur["tanf"], off_tanf=off["tanf"],
            cur_wic=cur["wic"], off_wic=off["wic"],
            cur_head_start=cur["head_start"], off_head_start=off["head_start"],
            cur_eitc=cur["eitc"], off_eitc=off["eitc"],
            cur_ssdi=cur["ssdi"], off_ssdi=off["ssdi"],
            cur_ssi=cur["ssi"], off_ssi=off["ssi"],
            cur_benefits_total=cur["benefits_total"],
            off_benefits_total=off["benefits_total"],
            delta_true_net_income=round(off["true_net_income"] - cur["true_net_income"], 2),
            delta_benefits_total=round(off["benefits_total"] - cur["benefits_total"], 2),
            delta_net_earnings=round(off["monthly_net_earnings"] - cur["monthly_net_earnings"], 2),
            is_cliff=0,
            programs_lost=0,
            programs_reduced=0,
        )
        rows.append(row)
        vignette_counter += 1

    # Generate 18 main vignettes
    for geometry, scenarios, target_label in [
        ("CLIFF", cliff_scenarios, "cliff"),
        ("PLATEAU", plateau_scenarios, "plateau"),
        ("POSITIVE", positive_scenarios, "positive")
    ]:
        for opp_level in ["low", "medium", "high"]:
            for rep in range(2):
                scenario = scenarios[rep % len(scenarios)]
                family = rng.choice(family_types)
                state, county = rng.choice(locations)

                current_wage = float(scenario["current_wage"]) + rng.uniform(-0.5, 0.5)
                offer_wage = float(scenario["offer_wage"]) + rng.uniform(-0.5, 0.5)
                hours = scenario["hours"]

                current_wage = max(10.0, current_wage)
                if offer_wage <= current_wage:
                    offer_wage = current_wage + rng.uniform(1.0, 2.0)

                location = LocationBenefits(
                    state=state, county=county,
                    num_children=family["num_children"],
                    child_ages=family["child_ages"],
                    has_disability=family["has_disability"]
                )

                cur = compute_bundle(current_wage, hours, location)
                off = compute_bundle(offer_wage, hours, location)

                delta_true = round(off["true_net_income"] - cur["true_net_income"], 2)
                delta_ben = round(off["benefits_total"] - cur["benefits_total"], 2)
                delta_netearn = round(off["monthly_net_earnings"] - cur["monthly_net_earnings"], 2)

                actual_type = "positive"
                if delta_netearn > 0 and delta_true < -50:
                    actual_type = "cliff"
                elif delta_netearn > 0 and -50 <= delta_true <= 100:
                    actual_type = "plateau"

                programs = ["snap", "medicaid", "ccdf", "tanf", "wic", "head_start", "eitc", "ssdi", "ssi"]
                programs_lost = sum([1 for prog in programs if cur[prog] > 0 and off[prog] == 0])
                programs_reduced = sum([1 for prog in programs if cur[prog] > off[prog] > 0])

                household_size = 1 + (1 if family["is_married"] else 0) + family["other_adults"] + family[
                    "num_children"]
                child_ages_str = ", ".join(str(age) for age in family["child_ages"]) if family[
                                                                                            "num_children"] > 0 else "none"

                row = dict(
                    vignette_id=vignette_counter,
                    scenario_type=actual_type,
                    target_type=target_label,
                    opportunity_level=opp_level,
                    current_job_desc=opportunity_descriptions[opp_level]["current_job"],
                    offer_job_desc=opportunity_descriptions[opp_level]["offer_job"],
                    mobility_description=opportunity_descriptions[opp_level]["mobility_text"],
                    state=state, county=county,
                    family_description=family["desc"],
                    num_children=family["num_children"],
                    has_disability=int(family["has_disability"]),
                    adult_age=family["adult_age"],
                    is_married=int(family["is_married"]),
                    spouse_age=family["spouse_age"] if family["spouse_age"] is not None else 0,
                    other_adults=family["other_adults"],
                    household_size=household_size,
                    child_ages_str=child_ages_str,
                    current_hourly_wage=round(current_wage, 2),
                    current_hours=int(hours),
                    offer_hourly_wage=round(offer_wage, 2),
                    offer_hours=int(hours),
                    cur_true_net_income=cur["true_net_income"],
                    off_true_net_income=off["true_net_income"],
                    cur_monthly_gross=cur["monthly_gross"],
                    off_monthly_gross=off["monthly_gross"],
                    cur_monthly_net_earnings=cur["monthly_net_earnings"],
                    off_monthly_net_earnings=off["monthly_net_earnings"],
                    cur_snap=cur["snap"], off_snap=off["snap"],
                    cur_medicaid=cur["medicaid"], off_medicaid=off["medicaid"],
                    cur_ccdf=cur["ccdf"], off_ccdf=off["ccdf"],
                    cur_tanf=cur["tanf"], off_tanf=off["tanf"],
                    cur_wic=cur["wic"], off_wic=off["wic"],
                    cur_head_start=cur["head_start"], off_head_start=off["head_start"],
                    cur_eitc=cur["eitc"], off_eitc=off["eitc"],
                    cur_ssdi=cur["ssdi"], off_ssdi=off["ssdi"],
                    cur_ssi=cur["ssi"], off_ssi=off["ssi"],
                    cur_benefits_total=cur["benefits_total"],
                    off_benefits_total=off["benefits_total"],
                    delta_true_net_income=delta_true,
                    delta_benefits_total=delta_ben,
                    delta_net_earnings=delta_netearn,
                    is_cliff=int(delta_netearn > 0 and delta_true < 0),
                    programs_lost=programs_lost,
                    programs_reduced=programs_reduced,
                )
                rows.append(row)
                vignette_counter += 1

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["vignette_id"] = i + 1

    path = vignette_csv_path()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_vignettes() -> list[dict]:
    path = vignette_csv_path()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in {"vignette_id", "is_cliff", "programs_lost", "programs_reduced", "num_children",
                         "has_disability", "adult_age", "is_married", "spouse_age", "other_adults", "household_size"}:
                    parsed[k] = int(float(v))
                elif k in {"family_description", "state", "county", "scenario_type", "target_type", "child_ages_str",
                           "opportunity_level", "current_job_desc", "offer_job_desc", "mobility_description"}:
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            out.append(parsed)
    return out


# ============================================
# oTree Models
# ============================================

class C(BaseConstants):
    NAME_IN_URL = "benefit_cliff"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Vignette identifiers
    vignette_id = models.IntegerField()
    scenario_type = models.StringField()
    target_type = models.StringField()
    opportunity_level = models.StringField()

    # Job descriptions
    current_job_desc = models.LongStringField()
    offer_job_desc = models.LongStringField()
    mobility_description = models.LongStringField()

    # Demographics
    state = models.StringField()
    county = models.StringField()
    family_description = models.StringField()
    num_children = models.IntegerField()
    has_disability = models.IntegerField()
    adult_age = models.IntegerField()
    is_married = models.IntegerField()
    spouse_age = models.IntegerField()
    other_adults = models.IntegerField()
    household_size = models.IntegerField()
    child_ages_str = models.StringField()

    # Job parameters
    current_hourly_wage = models.FloatField()
    current_hours = models.IntegerField()
    offer_hourly_wage = models.FloatField()
    offer_hours = models.IntegerField()

    # Current benefits
    cur_snap = models.FloatField()
    cur_medicaid = models.FloatField()
    cur_ccdf = models.FloatField()
    cur_tanf = models.FloatField()
    cur_wic = models.FloatField()
    cur_head_start = models.FloatField()
    cur_eitc = models.FloatField()
    cur_ssdi = models.FloatField()
    cur_ssi = models.FloatField()
    cur_benefits_total = models.FloatField()
    cur_true_net_income = models.FloatField()
    cur_monthly_gross = models.FloatField()
    cur_monthly_net_earnings = models.FloatField()

    # Offer benefits
    off_snap = models.FloatField()
    off_medicaid = models.FloatField()
    off_ccdf = models.FloatField()
    off_tanf = models.FloatField()
    off_wic = models.FloatField()
    off_head_start = models.FloatField()
    off_eitc = models.FloatField()
    off_ssdi = models.FloatField()
    off_ssi = models.FloatField()
    off_benefits_total = models.FloatField()
    off_true_net_income = models.FloatField()
    off_monthly_gross = models.FloatField()
    off_monthly_net_earnings = models.FloatField()

    # Deltas
    delta_true_net_income = models.FloatField()
    delta_net_earnings = models.FloatField()
    delta_benefits_total = models.FloatField()
    is_cliff = models.IntegerField()
    programs_lost = models.IntegerField()
    programs_reduced = models.IntegerField()

    # Treatment assignment
    treatment_format = models.StringField()

    # STAGE 1: Text baseline responses
    stage1_perceived_delta = models.FloatField(
        blank=True,
        min=-10000,
        max=10000
    )
    stage1_decision = models.IntegerField(
        choices=[[1, "Accept the offer"], [0, "Keep current job"]],
        widget=widgets.RadioSelect,
        blank=True
    )

    # STAGE 2: Treatment responses
    stage2_perceived_delta = models.FloatField(
        blank=True,
        min=-10000,
        max=10000
    )
    stage2_decision = models.IntegerField(
        choices=[[1, "Accept the offer"], [0, "Keep current job"]],
        widget=widgets.RadioSelect,
        blank=True
    )

    # STAGE 3: Reconsideration responses
    final_decision = models.IntegerField(
        choices=[[1, "Accept the offer"], [0, "Keep current job"]],
        blank=True
    )

    changed_mind = models.BooleanField(blank=True)

    # Q3: Main reason (SIMPLIFIED - single choice)
    main_reason = models.StringField(
        choices=[
            ['more_money', 'The new job offers more total monthly resources'],
            ['same_money', 'Both options offer about the same monthly resources'],
            ['future_opportunities', 'Future career growth and advancement opportunities'],
            ['higher_salary', 'Higher salary is more important than benefits'],
            ['benefits_matter', 'Keeping my current benefits is more important'],
            ['other', 'Other reason'],
        ],
        blank=True
    )

    reason_other_text = models.LongStringField(blank=True)

    # Q4: Future expectations
    future_expectation = models.IntegerField(
        choices=[
            [1, "Much worse (>$5,000/year less)"],
            [2, "Somewhat worse ($1,000-$5,000/year less)"],
            [3, "About the same"],
            [4, "Somewhat better ($1,000-$5,000/year more)"],
            [5, "Much better (>$5,000/year more)"],
        ],
        blank=True
    )

    # Q5: Willingness to sacrifice
    max_acceptable_loss = models.IntegerField(
        choices=[
            [0, "$0 (I would not accept any loss)"],
            [50, "Up to $50/month"],
            [100, "Up to $100/month"],
            [150, "Up to $150/month"],
            [200, "Up to $200/month"],
            [999, "More than $200/month"],
        ],
        blank=True
    )


def creating_session(subsession: Subsession):
    if not vignette_csv_path().exists():
        generate_vignettes_csv(N_VIGNETTES, SEED)

    vignettes = load_vignettes()
    players = subsession.get_players()
    rng = random.Random(SEED + subsession.session.config.get('random_seed', 0))

    n_players = len(players)
    repetitions_per_vignette = (n_players // len(vignettes)) + 1

    assignments = []
    for _ in range(repetitions_per_vignette):
        assignments.extend(vignettes)

    rng.shuffle(assignments)
    assignments = assignments[:n_players]

    for p, v in zip(players, assignments[:n_players]):
        p.vignette_id = v["vignette_id"]
        p.scenario_type = v["scenario_type"]
        p.target_type = v["target_type"]
        p.opportunity_level = v["opportunity_level"]
        p.current_job_desc = v["current_job_desc"]
        p.offer_job_desc = v["offer_job_desc"]
        p.mobility_description = v["mobility_description"]
        p.state = v["state"]
        p.county = v["county"]
        p.family_description = v["family_description"]
        p.num_children = v["num_children"]
        p.has_disability = v["has_disability"]
        p.adult_age = v["adult_age"]
        p.is_married = v["is_married"]
        p.spouse_age = v["spouse_age"]
        p.other_adults = v["other_adults"]
        p.household_size = v["household_size"]
        p.child_ages_str = v["child_ages_str"]
        p.current_hourly_wage = v["current_hourly_wage"]
        p.current_hours = v["current_hours"]
        p.offer_hourly_wage = v["offer_hourly_wage"]
        p.offer_hours = v["offer_hours"]

        p.cur_snap = v["cur_snap"]
        p.cur_medicaid = v["cur_medicaid"]
        p.cur_ccdf = v["cur_ccdf"]
        p.cur_tanf = v["cur_tanf"]
        p.cur_wic = v["cur_wic"]
        p.cur_head_start = v["cur_head_start"]
        p.cur_eitc = v["cur_eitc"]
        p.cur_ssdi = v["cur_ssdi"]
        p.cur_ssi = v["cur_ssi"]
        p.cur_benefits_total = v["cur_benefits_total"]
        p.cur_true_net_income = v["cur_true_net_income"]
        p.cur_monthly_gross = v["cur_monthly_gross"]
        p.cur_monthly_net_earnings = v["cur_monthly_net_earnings"]

        p.off_snap = v["off_snap"]
        p.off_medicaid = v["off_medicaid"]
        p.off_ccdf = v["off_ccdf"]
        p.off_tanf = v["off_tanf"]
        p.off_wic = v["off_wic"]
        p.off_head_start = v["off_head_start"]
        p.off_eitc = v["off_eitc"]
        p.off_ssdi = v["off_ssdi"]
        p.off_ssi = v["off_ssi"]
        p.off_benefits_total = v["off_benefits_total"]
        p.off_true_net_income = v["off_true_net_income"]
        p.off_monthly_gross = v["off_monthly_gross"]
        p.off_monthly_net_earnings = v["off_monthly_net_earnings"]

        p.delta_true_net_income = v["delta_true_net_income"]
        p.delta_net_earnings = v["delta_net_earnings"]
        p.delta_benefits_total = v["delta_benefits_total"]
        p.is_cliff = v["is_cliff"]
        p.programs_lost = v["programs_lost"]
        p.programs_reduced = v["programs_reduced"]

        p.treatment_format = rng.choice(["dashboard", "chatgpt"])


# ============================================
# PAGES
# ============================================

class WelcomeFromQualtrics(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def before_next_page(player, timeout_happened):
        participant = player.participant
        session = player.session

        prolific_pid = session.config.get('prolific_pid', 'TEST_NO_ID')
        qualtrics_id = session.config.get('qualtrics_id', 'TEST_NO_ID')

        participant.prolific_pid = prolific_pid
        participant.qualtrics_response_id = qualtrics_id

        completion_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        participant.completion_code = completion_code


class Stage1TextPage(Page):
    form_model = "player"
    form_fields = ["stage1_perceived_delta", "stage1_decision"]

    @staticmethod
    def error_message(player, values):
        # Allow negative values - no validation errors
        pass

    @staticmethod
    def vars_for_template(player: Player):
        cur_gross = player.current_hourly_wage * player.current_hours * WEEKS_PER_MONTH
        off_gross = player.offer_hourly_wage * player.offer_hours * WEEKS_PER_MONTH

        return dict(
            current_wage=player.current_hourly_wage,
            current_hours=player.current_hours,
            current_monthly_gross=round(cur_gross, 2),
            current_annual_gross=round(cur_gross * 12, 2),
            offer_monthly_gross=round(off_gross, 2),
            offer_annual_gross=round(off_gross * 12, 2),
            show_benefits=True,
            show_ccdf=player.num_children > 0,
            show_tanf=player.num_children > 0,
            show_wic=player.num_children > 0,
            show_head_start=player.num_children > 0,
            show_disability=player.has_disability == 1,
            opportunity_level=player.opportunity_level,
            current_job_desc=player.current_job_desc,
            offer_job_desc=player.offer_job_desc,
            mobility_desc=player.mobility_description,
        )


class Stage2TreatmentPage(Page):
    form_model = "player"
    form_fields = ["stage2_perceived_delta", "stage2_decision"]

    @staticmethod
    def error_message(player, values):
        # Allow negative values - no validation errors
        pass

    @staticmethod
    def vars_for_template(player: Player):
        cur_gross = player.current_hourly_wage * player.current_hours * WEEKS_PER_MONTH
        off_gross = player.offer_hourly_wage * player.offer_hours * WEEKS_PER_MONTH

        return dict(
            current_wage=player.current_hourly_wage,
            current_hours=player.current_hours,
            current_monthly_gross=round(cur_gross, 2),
            current_annual_gross=round(cur_gross * 12, 2),
            offer_monthly_gross=round(off_gross, 2),
            offer_annual_gross=round(off_gross * 12, 2),
            show_benefits=True,
            show_ccdf=player.num_children > 0,
            show_tanf=player.num_children > 0,
            show_wic=player.num_children > 0,
            show_head_start=player.num_children > 0,
            show_disability=player.has_disability == 1,
            opportunity_level=player.opportunity_level,
            current_job_desc=player.current_job_desc,
            offer_job_desc=player.offer_job_desc,
            mobility_desc=player.mobility_description,
            treatment_format=player.treatment_format,
            delta_abs=abs(player.delta_true_net_income),
        )


class FeedbackPage(Page):
    @staticmethod
    def vars_for_template(player: Player):
        cur_gross = player.current_hourly_wage * player.current_hours * WEEKS_PER_MONTH
        off_gross = player.offer_hourly_wage * player.offer_hours * WEEKS_PER_MONTH

        stage1_error = None
        stage2_error = None
        stage1_error_abs = None
        stage2_error_abs = None

        if player.stage1_perceived_delta is not None:
            stage1_error = round(player.stage1_perceived_delta - player.delta_true_net_income, 2)
            stage1_error_abs = abs(stage1_error)

        if player.stage2_perceived_delta is not None:
            stage2_error = round(player.stage2_perceived_delta - player.delta_true_net_income, 2)
            stage2_error_abs = abs(stage2_error)

        # Calculate benefit changes for template
        delta_snap = round(player.off_snap - player.cur_snap, 2)
        delta_medicaid = round(player.off_medicaid - player.cur_medicaid, 2)
        delta_ccdf = round(player.off_ccdf - player.cur_ccdf, 2)
        delta_tanf = round(player.off_tanf - player.cur_tanf, 2)
        delta_wic = round(player.off_wic - player.cur_wic, 2)
        delta_head_start = round(player.off_head_start - player.cur_head_start, 2)
        delta_eitc = round(player.off_eitc - player.cur_eitc, 2)
        delta_ssdi = round(player.off_ssdi - player.cur_ssdi, 2)
        delta_ssi = round(player.off_ssi - player.cur_ssi, 2)

        return dict(
            current_monthly_gross=round(cur_gross, 2),
            offer_monthly_gross=round(off_gross, 2),
            show_ccdf=player.num_children > 0,
            show_tanf=player.num_children > 0,
            show_wic=player.num_children > 0,
            show_head_start=player.num_children > 0,
            show_disability=player.has_disability == 1,
            stage1_error=stage1_error,
            stage2_error=stage2_error,
            stage1_error_abs=stage1_error_abs,
            stage2_error_abs=stage2_error_abs,
            treatment_format=player.treatment_format,
            delta_abs=abs(player.delta_true_net_income),
            delta_snap=delta_snap,
            delta_medicaid=delta_medicaid,
            delta_ccdf=delta_ccdf,
            delta_tanf=delta_tanf,
            delta_wic=delta_wic,
            delta_head_start=delta_head_start,
            delta_eitc=delta_eitc,
            delta_ssdi=delta_ssdi,
            delta_ssi=delta_ssi,
            current_job_desc=player.current_job_desc,
            offer_job_desc=player.offer_job_desc,
            mobility_desc=player.mobility_description,
        )


class ReconsiderationPage(Page):
    form_model = "player"
    form_fields = [
        "changed_mind",
        "final_decision",
        "main_reason",
        "reason_other_text",
        "future_expectation",
        "max_acceptable_loss"
    ]

    @staticmethod
    def vars_for_template(player: Player):
        return dict(
            stage1_decision_text="Accepted" if player.stage1_decision == 1 else "Rejected",
            stage2_decision_text="Accepted" if player.stage2_decision == 1 else "Rejected",
        )


class FinalPageProlific(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player):
        completion_code = player.participant.completion_code
        return {
            'completion_code': completion_code,
            'prolific_completion_url': f'https://app.prolific.com/submissions/complete?cc={completion_code}'
        }


page_sequence = [
    WelcomeFromQualtrics,
    Stage1TextPage,
    Stage2TreatmentPage,
    FeedbackPage,
    ReconsiderationPage,
    FinalPageProlific,
]
# Force deploy Sun Jan 19 04:45:58 EST 2026