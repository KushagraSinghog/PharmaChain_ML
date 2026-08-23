import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import DBSCAN
from app.config import settings
from app.features.geo import haversine_distance_km

class ComplaintClusterer:
    """
    Geospatial and temporal clustering using scikit-learn DBSCAN.
    Identifies geographic hotspots of consumer/inspector counterfeit complaints.
    """

    def __init__(
        self,
        eps_km: float = settings.DBSCAN_EPS_KM,
        min_samples: int = settings.DBSCAN_MIN_SAMPLES
    ):
        self.eps_km = eps_km
        self.min_samples = min_samples
        # Convert eps to radians on earth sphere for haversine metric
        self.eps_radians = eps_km / settings.EARTH_RADIUS_KM

    def cluster_reports(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs DBSCAN on report latitude/longitude coordinates.
        Returns detected clusters with centroids, participating entities, and severity.
        """
        valid_reports = [
            r for r in reports
            if r.get("latitude") is not None and r.get("longitude") is not None
            and (r.get("latitude") != 0.0 or r.get("longitude") != 0.0)
        ]

        if len(valid_reports) < self.min_samples:
            return {
                "total_reports": len(reports),
                "clustered_reports_count": 0,
                "noise_count": len(valid_reports),
                "clusters_count": 0,
                "clusters": []
            }

        # Coordinates matrix in radians [lat, lon]
        coords = np.array([
            [np.radians(r["latitude"]), np.radians(r["longitude"])]
            for r in valid_reports
        ])

        db = DBSCAN(
            eps=self.eps_radians,
            min_samples=self.min_samples,
            metric="haversine",
            algorithm="ball_tree"
        )
        labels = db.fit_predict(coords)

        clusters_dict: Dict[int, List[Dict[str, Any]]] = {}
        noise_reports: List[Dict[str, Any]] = []

        for idx, label in enumerate(labels):
            report_item = valid_reports[idx]
            if label == -1:
                noise_reports.append(report_item)
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(report_item)

        clusters_output = []
        for cluster_label, items in clusters_dict.items():
            lats = [it["latitude"] for it in items]
            lons = [it["longitude"] for it in items]
            centroid_lat = float(np.mean(lats))
            centroid_lon = float(np.mean(lons))

            # Max distance from centroid
            radius_km = max([
                haversine_distance_km(centroid_lat, centroid_lon, it["latitude"], it["longitude"])
                for it in items
            ]) if items else 0.0

            shops = list(set(it["shop_id"] for it in items if it.get("shop_id")))
            batches = list(set(it["batch_id"] for it in items if it.get("batch_id")))
            types = {}
            for it in items:
                rtype = it.get("report_type", "UNKNOWN")
                types[rtype] = types.get(rtype, 0) + 1

            size = len(items)
            risk = "HIGH" if size < 6 else "CRITICAL"

            clusters_output.append({
                "cluster_id": f"CLUSTER-{cluster_label + 1:03d}",
                "size": size,
                "risk_level": risk,
                "centroid": {
                    "latitude": round(centroid_lat, 5),
                    "longitude": round(centroid_lon, 5)
                },
                "radius_km": round(radius_km, 2),
                "participating_shops": shops,
                "participating_batches": batches,
                "report_types_breakdown": types,
                "reports": [
                    {
                        "report_id": it.get("report_id"),
                        "batch_id": it.get("batch_id"),
                        "shop_id": it.get("shop_id"),
                        "report_type": it.get("report_type"),
                        "latitude": it.get("latitude"),
                        "longitude": it.get("longitude"),
                        "timestamp": it.get("timestamp").isoformat() if hasattr(it.get("timestamp"), "isoformat") else str(it.get("timestamp"))
                    }
                    for it in items
                ]
            })

        # Sort clusters by size descending
        clusters_output.sort(key=lambda x: x["size"], reverse=True)

        return {
            "total_reports": len(reports),
            "clustered_reports_count": sum(c["size"] for c in clusters_output),
            "noise_count": len(noise_reports),
            "clusters_count": len(clusters_output),
            "clusters": clusters_output
        }

clusterer = ComplaintClusterer()
