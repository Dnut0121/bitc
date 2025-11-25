from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="dashboard-index"),
    path("api/candles/", views.candles_api, name="dashboard-candles-api"),
]

