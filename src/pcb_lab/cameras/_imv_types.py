"""Native IMV ABI subset, checked against Huaray SDK 2.4.1 IMVDefines.h.

Windows x64, default C alignment. Do not use the unrelated MindVision SDK.
"""
import ctypes as C

U32, I32, U64 = C.c_uint32, C.c_int32, C.c_uint64
String = C.c_char * 256


class GigEDevice(C.Structure):
    _fields_ = [("options", U32), ("current", U32), ("reserved", U32 * 3)] + [
        (name, String) for name in ("mac", "ip", "subnet", "gateway", "protocol", "configuration")
    ] + [("strings_reserved", String * 6)]


class UsbDevice(C.Structure):
    _fields_ = [("flags", C.c_bool * 8), ("reserved", U32 * 4), ("strings", String * 12)]


class DeviceSpecific(C.Union):
    _fields_ = [("gige", GigEDevice), ("usb", UsbDevice)]


class GigEInterface(C.Structure):
    _fields_ = [(name, String) for name in ("description", "mac", "ip", "subnet", "gateway")] + [
        ("reserved", String * 5)]


class InterfaceSpecific(C.Union):
    _fields_ = [("gige", GigEInterface), ("usb", String * 10)]


class DeviceInfo(C.Structure):
    _fields_ = [("camera_type", I32), ("reserved", I32 * 5)] + [
        (name, String) for name in ("key", "name", "serial", "vendor", "model", "manufacture", "version")
    ] + [("camera_reserved", String * 5), ("device", DeviceSpecific),
         ("interface_type", I32), ("interface_reserved", I32 * 5),
         ("interface_name", String), ("interface_strings", String * 5), ("interface", InterfaceSpecific)]


class DeviceList(C.Structure):
    _fields_ = [("count", U32), ("devices", C.POINTER(DeviceInfo))]


class FrameInfo(C.Structure):
    _fields_ = [("block_id", U64), ("status", U32), ("width", U32), ("height", U32),
                ("size", U32), ("pixel_format", I32), ("timestamp", U64),
                ("chunk_count", U32), ("padding_x", U32), ("padding_y", U32),
                ("receive_time", U32), ("reserved", U32 * 19)]


class Frame(C.Structure):
    _fields_ = [("handle", C.c_void_p), ("data", C.POINTER(C.c_ubyte)),
                ("info", FrameInfo), ("reserved", U32 * 10)]


def bind(dll):
    handle, text = C.c_void_p, C.c_char_p
    signatures = {
        "GetVersion": (text, []),
        "EnumDevices": (I32, [C.POINTER(DeviceList), U32]),
        "CreateHandle": (I32, [C.POINTER(handle), I32, handle]),
        "DestroyHandle": (I32, [handle]), "Open": (I32, [handle]), "Close": (I32, [handle]),
        "StartGrabbing": (I32, [handle]), "StopGrabbing": (I32, [handle]),
        "GetFrame": (I32, [handle, C.POINTER(Frame), U32]),
        "ReleaseFrame": (I32, [handle, C.POINTER(Frame)]),
        "GetEnumFeatureValue": (I32, [handle, text, C.POINTER(U64)]),
        "SetEnumFeatureValue": (I32, [handle, text, U64]),
        "SetEnumFeatureSymbol": (I32, [handle, text, text]),
        "GetDoubleFeatureValue": (I32, [handle, text, C.POINTER(C.c_double)]),
        "SetDoubleFeatureValue": (I32, [handle, text, C.c_double]),
        "ExecuteCommandFeature": (I32, [handle, text]),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(dll, "IMV_" + name)
        fn.restype, fn.argtypes = result, args
    return dll
