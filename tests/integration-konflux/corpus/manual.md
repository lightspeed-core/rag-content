---
url: https://docs.quorbitex.example.com/zyranex-t7/manual
title: "Zyranex Model T7 — Product Manual"
---

# Zyranex Model T7 — Product Manual

## Quorbitex Industries — Confidential

### 1. Overview

The Zyranex Model T7 is a plasma-cooled quantum relay manufactured by Quorbitex
Industries. It is designed for sub-orbital mesh networking deployments and
operates at frequencies between 47.3 GHz and 91.7 GHz. The T7 is the successor
to the Zyranex Model T5 and introduces the Quorbitex Reliability Protocol (QRP)
for automatic failover in multi-relay configurations.

### 2. Specifications

| Parameter | Value |
|-----------|-------|
| Model | Zyranex T7 |
| Manufacturer | Quorbitex Industries |
| Cooling | Plasma-cooled (Quorbitex PCX-3 system) |
| Operating frequency | 47.3–91.7 GHz |
| Max relay hops | 12 (with QRP enabled) |
| Power consumption | 340 W nominal, 510 W peak |
| Dimensions | 42 × 28 × 15 cm |
| Weight | 4.7 kg |
| Certification | Quorbitex QRP Level 3 |

### 3. Installation

1. Mount the Zyranex T7 unit using the supplied Quorbitex Universal Bracket
   (QUB-200).
2. Connect the PCX-3 plasma coolant line to the rear port labeled "COOLANT IN".
3. Attach the sub-orbital antenna array to the top connector.
4. Power on and wait for the QRP initialization sequence (approximately 45
   seconds). The front panel LED will change from amber to green when the
   Phase-Lock Calibration Sequence completes.
5. Configure the mesh network ID using the Quorbitex Management Console (QMC)
   software.

### 4. Troubleshooting

| Error Code | Description | Resolution |
|------------|-------------|------------|
| ZRX-4401 | Plasma coolant pressure below threshold | Check PCX-3 coolant reservoir level. Refill with Quorbitex-approved PCX fluid. |
| ZRX-4402 | Phase-Lock Calibration failure | Power cycle the unit. If error persists, replace the oscillator module (part QOC-77). |
| ZRX-7710 | QRP failover loop detected | Reduce relay hop count below 12 or update firmware to v3.8+. |

### 5. Warranty

The Zyranex Model T7 is covered by the Quorbitex Industries Standard Equipment
Warranty for a period of 36 months from the date of purchase. This warranty
covers defects in materials and workmanship under normal use. Plasma coolant
consumables (PCX fluid) are not covered. Contact Quorbitex Support at
support@quorbitex.example.com for warranty claims.
