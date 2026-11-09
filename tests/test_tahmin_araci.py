import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tahmin_araci as t  # noqa: E402


@pytest.fixture(scope="module")
def yuklu():
    return t.yukle()


def test_risk_sinirlari(yuklu):
    r = t.risk_hesapla({"failures": 2, "studytime": 1, "absences": 14}, *yuklu)
    assert 0.0 <= r <= 1.0


def test_bos_girdi_varsayilanla_calisir(yuklu):
    assert 0.0 <= t.risk_hesapla({}, *yuklu) <= 1.0


def test_riskli_profil_daha_yuksek(yuklu):
    riskli = t.risk_hesapla({"failures": 3, "studytime": 1, "absences": 30, "higher": "no", "goout": 5}, *yuklu)
    guvenli = t.risk_hesapla({"failures": 0, "studytime": 4, "absences": 0, "higher": "yes", "goout": 1}, *yuklu)
    assert riskli > guvenli


@pytest.mark.parametrize("kotu", [{"failures": 9}, {"higher": "belki"}, {"bilinmeyen": 1}, {"studytime": 0}])
def test_gecersiz_girdi_reddedilir(kotu, yuklu):
    with pytest.raises(ValueError):
        t.risk_hesapla(kotu, *yuklu)


def test_yorum_seviyeleri():
    assert "YÜKSEK" in t.yorumla(0.8) and "ORTA" in t.yorumla(0.5) and "düşük" in t.yorumla(0.1)


def test_cli_cikis_kodlari(capsys):
    assert t.main(["--failures", "1"]) == 0
    assert t.main(["--failures", "99"]) == 2
