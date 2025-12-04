from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="dashboard-index"),
    path("api/candles/", views.candles_api, name="dashboard-candles-api"),
    path("api/hybrid-live/", views.hybrid_live_api, name="dashboard-hybrid-live-api"),
]

