"""Reference-specific geometry checks for this diagnostic, not a general sofa metric."""
import math


def check_sofa(measurements, original=None):
    errors = []
    expected = {'base_frame': 1, 'legs': 4, 'armrests': 2, 'seat_cushions': 3, 'back_cushions': 3}
    for name, count in expected.items():
        components = measurements.get(name, {}).get('components', [])
        if len(components) != count:
            errors.append(f'{name}: expected {count} connected components; found {len(components)}')
        for c in components:
            if not all(math.isfinite(v) for v in c['min']+c['max']) or min(c['dimensions']) <= 0:
                errors.append(f'{name}: invalid geometry bounds')
    if original:
        for name in ('base_frame', 'legs'):
            if measurements.get(name, {}).get('geometry_sha256') != original[name]['geometry_sha256']:
                errors.append(f'{name}: protected geometry changed')
    arms = sorted(measurements.get('armrests', {}).get('components', []), key=lambda c:c['center'][0])
    seats = sorted(measurements.get('seat_cushions', {}).get('components', []), key=lambda c:c['center'][0])
    backs = sorted(measurements.get('back_cushions', {}).get('components', []), key=lambda c:c['center'][0])
    if len(arms) == 2:
        if not (arms[0]['center'][0] < -.35 and arms[1]['center'][0] > .35):
            errors.append('Armrests must occupy opposite outer sides')
        if abs(arms[0]['center'][0]+arms[1]['center'][0]) > .01:
            errors.append('Armrest positions are asymmetric')
        for c in arms:
            if c['max'][2] < .35:
                errors.append('Armrest top is too low relative to the seat')
        for c in seats:
            if c['min'][0] < arms[0]['max'][0]-.012 or c['max'][0] > arms[1]['min'][0]+.012:
                errors.append('Seat cushions significantly overlap the side armrests')
    for c in backs:
        if c['dimensions'][0] < .21 or c['dimensions'][2] < .22:
            errors.append('Back cushion is too small; expected a broad full-height cushion')
        if c['min'][2] > .34 or c['max'][2] < .46:
            errors.append('Back cushion does not span the seat-to-back region')
        if c['dimensions'][0] < .9*c['dimensions'][2]:
            errors.append('Back cushion is too tall relative to its width')
    if len(backs) == len(seats) == 3:
        for back, seat in zip(backs, seats):
            if abs(back['center'][0]-seat['center'][0]) > .025:
                errors.append('Back and seat cushion centers do not align')
            if back['center'][1] <= seat['center'][1]:
                errors.append('Back cushion is not behind its seat')
            if back['max'][2] - seat['max'][2] < .17:
                errors.append('Too little back cushion is exposed above the seat (less than 0.17 units)')
    # AABB contact is a necessary coarse connectivity check, not surface proof.
    components = [dict(c, group=g) for g,m in measurements.items() for c in m['components']]
    if components:
        width=max(c['max'][0] for c in components)-min(c['min'][0] for c in components)
        height=max(c['max'][2] for c in components)-min(c['min'][2] for c in components)
        if height > .55*width:
            errors.append('Overall sofa height-to-width ratio exceeds 0.55 for this reference')
    reached = {i for i,c in enumerate(components) if c['group']=='base_frame'}
    for _ in components:
        for i,a in enumerate(components):
            for j in tuple(reached):
                b=components[j]
                gap=math.sqrt(sum(max(0,a['min'][k]-b['max'][k],b['min'][k]-a['max'][k])**2 for k in range(3)))
                if gap <= .008:
                    reached.add(i)
    if len(reached) != len(components):
        errors.append('One or more parts are disconnected from the base under the 8mm AABB check')
    return {'passed': not errors, 'errors': sorted(set(errors)),
            'limits': 'Reference-specific normalized-coordinate checks. AABB connectivity and counts do not prove shape fidelity; independent visual review is required.'}
