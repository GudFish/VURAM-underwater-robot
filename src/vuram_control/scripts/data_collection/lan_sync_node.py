#!/usr/bin/env python3
"""Run LAN upload/model synchronization separately from the control process."""
import sys
import threading
from pathlib import Path

import rospy


def load_client_type(learning_root):
    """Reject an already imported learner from a different installation."""
    root = Path(learning_root).expanduser().resolve()
    if not (root / 'auv_learning').is_dir():
        raise ValueError('learning_root must contain auv_learning: {}'.format(root))
    sys.path.insert(0, str(root))
    import auv_learning
    if Path(auv_learning.__file__).resolve().parents[1] != root:
        raise ValueError('auv_learning was already imported from a different learning_root')
    from auv_learning.deployment.lan_client import LanClient
    return LanClient


def main():
    rospy.init_node('lan_sync', anonymous=False)
    stop = threading.Event()
    rospy.on_shutdown(stop.set)
    try:
        mode = rospy.get_param('~transport_mode', 'lan')
        if mode != 'lan':
            raise ValueError('lan_sync requires transport_mode=lan; got {!r}'.format(mode))
        root = Path(rospy.get_param('~learning_root', '~/mujoco_auv_env')).expanduser()
        client_type = load_client_type(root)
        client = client_type(
            server_url=rospy.get_param('~server_url', 'http://192.168.1.1:8765'),
            outbox=rospy.get_param('~data_dir', str(root / 'data/vnd/lan/outbox')),
            model_root=rospy.get_param('~model_root', str(root / 'data/vnd/lan/models')),
            robot_id=rospy.get_param('~robot_id', 'auv01'),
            token_file=rospy.get_param('~token_file', str(root / 'data/vnd/lan/token')),
            poll_interval=rospy.get_param('~poll_interval', 1.0),
            download_models=rospy.get_param('~download_models', True),
            upload_data=rospy.get_param('~upload_data', True),
            log=rospy.loginfo,
        )
        client.run(stop)
        return 0
    except Exception as exc:
        rospy.logerr('LAN synchronization failed: %s', exc)
        return 1
    finally:
        stop.set()


if __name__ == '__main__':
    sys.exit(main())
