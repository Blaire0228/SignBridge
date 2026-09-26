import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:model_viewer_plus/model_viewer_plus.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:webview_flutter/webview_flutter.dart';

import 'chat_page.dart';
import 'chat_session.dart';
import 'data_contribution_page.dart';

void main() {
  runApp(const TslTranslatorApp());
}

class TslTranslatorApp extends StatelessWidget {
  const TslTranslatorApp({super.key, this.enableModelViewer = true});

  final bool enableModelViewer;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'SignBridge',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF4D8DFF),
          surface: const Color(0xFFF8F8F8),
        ),
        scaffoldBackgroundColor: const Color(0xFFF8F8F8),
        useMaterial3: true,
      ),
      home: TranslatorHomePage(enableModelViewer: enableModelViewer),
    );
  }
}

enum TranslationMode { textToSign, signToText }

enum AnimationPlaybackState { idle, playing, paused, completed }

class TranslationRecord {
  const TranslationRecord({required this.source, required this.tsl});

  final String source;
  final String tsl;
}

class AnimationWordTiming {
  const AnimationWordTiming({
    required this.word,
    required this.start,
    required this.end,
  });

  final String word;
  final double start;
  final double end;
}

class RecognitionCandidate {
  const RecognitionCandidate({required this.label, required this.confidence});

  final String label;
  final double confidence;
}

class TranslatorHomePage extends StatefulWidget {
  const TranslatorHomePage({super.key, this.enableModelViewer = true});

  final bool enableModelViewer;

  @override
  State<TranslatorHomePage> createState() => _TranslatorHomePageState();
}

class _TranslatorHomePageState extends State<TranslatorHomePage>
    with WidgetsBindingObserver {
  static const _viewerGray = Color.fromARGB(255, 198, 198, 198);
  static const _cameraOrbit = '0deg 75deg 55%';
  // A slightly lower camera target places the avatar higher in the viewport.
  static const _cameraTarget = '0m 1.00m 0m';
  static const _minInputSheetExtent = 0.22;
  static const _resultMinInputSheetExtent = 0.22;
  static const _maxInputSheetExtent = 0.72;
  static const _resultInputSheetExtent = 0.27;
  static const _expandedInputSheetThreshold = 0.34;
  static const _inputTextSize = 16.0;
  static const _apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8000',
  );
  final TextEditingController _controller = TextEditingController();
  final TextEditingController _recognitionController = TextEditingController();
  final TextEditingController _chatTextController = TextEditingController();
  final FocusNode _textInputFocusNode = FocusNode();
  final DraggableScrollableController _inputSheetController =
      DraggableScrollableController();
  final DraggableScrollableController _signInputSheetController =
      DraggableScrollableController();
  final ScrollController _chatMessagesController = ScrollController();
  int _pendingWordTasks = 0;
  final List<TranslationRecord> _records = [];
  final SpeechToText _speechToText = SpeechToText();
  final FlutterTts _flutterTts = FlutterTts();

  TranslationMode _mode = TranslationMode.textToSign;
  CameraController? _cameraController;
  List<CameraDescription> _cameras = const [];
  int _selectedCameraIndex = 0;
  bool _isCameraInitializing = false;
  bool _isStartingSignRecording = false;
  bool _isRecordingSign = false;
  bool _isRecognizingSign = false;
  bool _usingRealtimeStream = false;
  bool _isSendingRealtimeFrame = false;
  String? _realtimeSessionId;
  DateTime? _lastRealtimeFrameAt;
  String? _recognizedSign;
  List<RecognitionCandidate> _recognitionCandidates = const [];
  String? _recognitionMessage;
  bool _isRefiningSentence = false;
  final List<String> _confirmedSignWords = [];
  final List<List<RecognitionCandidate>> _confirmedSignCandidates = [];
  String? _tslResult;
  String? _translationError;
  String? _glbUrl;
  String? _modelViewerUrl;
  String? _modelLoadError;
  final Set<File> _cachedModelFiles = {};
  bool _isTranslating = false;
  bool _speechInitialized = false;
  bool _isListening = false;
  bool _hasEnteredSentence = false;
  bool _isDataContributionPageOpen = false;
  String? _taiwanChineseLocaleId;
  double _playbackSpeed = 1;
  int _viewerRevision = 0;
  bool _autoPlayAnimation = true;
  bool _autoPlaySpeech = true;
  double _inputSheetExtent = _minInputSheetExtent;
  List<String> _animationWords = const [];
  List<AnimationWordTiming> _animationWordTimings = const [];
  int? _activeAnimationWordIndex;
  AnimationPlaybackState _animationPlaybackState = AnimationPlaybackState.idle;
  WebViewController? _modelViewerController;
  ChatWorkspaceConfig? _chatWorkspace;
  bool _chatWorkspaceVisible = false;
  double _chatSheetExtent = 0;
  int _lastObservedChatMessageCount = 0;

  String get _normalizedApiBaseUrl =>
      _apiBaseUrl.replaceFirst(RegExp(r'/+$'), '');

  Uri _apiUri(String path) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$_normalizedApiBaseUrl$normalizedPath');
  }

  String _resolveBackendUrl(String value) {
    final uri = Uri.tryParse(value);
    if (uri != null && uri.hasScheme) return value;
    return _apiUri(value).toString();
  }

  String get _idleGlbUrl => _apiUri('/actions/idle.glb').toString();
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    ChatSession.instance.addListener(_handleChatSessionChanged);
    unawaited(_initializeTts());
    if (widget.enableModelViewer) unawaited(_loadInitialModel());
  }

  void _handleChatSessionChanged() {
    if (!mounted) return;
    final session = ChatSession.instance;
    final previousCount = _lastObservedChatMessageCount.clamp(
      0,
      session.messages.length,
    );
    final incomingMessages = session.messages
        .skip(previousCount)
        .where((message) => message.senderId != session.deviceId)
        .toList(growable: false);
    _lastObservedChatMessageCount = session.messages.length;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      setState(() {
        if (!session.isActive) {
          _chatWorkspaceVisible = false;
          _chatWorkspace = null;
        } else if (_chatWorkspaceVisible && incomingMessages.isNotEmpty) {
          var requiredExtent = _chatSheetExtent;
          for (final message in incomingMessages) {
            final messageExtent =
                message.animationUrl.isNotEmpty || message.tsl.isNotEmpty
                ? 0.62
                : message.text.length > 100
                ? 0.48
                : message.text.length > 40
                ? 0.40
                : 0.32;
            if (messageExtent > requiredExtent) {
              requiredExtent = messageExtent;
            }
          }
          _chatSheetExtent = requiredExtent;
        }
      });
      if (_chatWorkspaceVisible && incomingMessages.isNotEmpty) {
        session.setChatVisible(true);
      }
      if (_chatMessagesController.hasClients) {
        _chatMessagesController.jumpTo(
          _chatMessagesController.position.maxScrollExtent,
        );
      }
    });
  }

  Future<String> _downloadModelForViewer(String modelUrl) async {
    final response = await http
        .get(
          Uri.parse(modelUrl),
          headers: const {'ngrok-skip-browser-warning': 'true'},
        )
        .timeout(const Duration(minutes: 2));

    if (response.statusCode != 200) {
      throw HttpException(
        'Model download failed with HTTP ${response.statusCode}',
        uri: Uri.parse(modelUrl),
      );
    }
    if (response.bodyBytes.length < 4 ||
        ascii.decode(response.bodyBytes.sublist(0, 4), allowInvalid: true) !=
            'glTF') {
      throw const FormatException('Model response is not a GLB file');
    }

    final file = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}'
      'tsl_model_${DateTime.now().microsecondsSinceEpoch}.glb',
    );
    await file.writeAsBytes(response.bodyBytes, flush: true);
    _cachedModelFiles.add(file);
    return file.uri.toString();
  }

  Future<void> _loadInitialModel() async {
    try {
      final localUrl = await _downloadModelForViewer(_idleGlbUrl);
      if (!mounted) return;
      setState(() {
        _modelViewerUrl = localUrl;
        _modelLoadError = null;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _modelLoadError = error.toString());
    }
  }

  Future<void> _initializeTts() async {
    try {
      final engines = await _flutterTts.getEngines;
      if (engines is List && engines.contains('com.google.android.tts')) {
        await _flutterTts.setEngine('com.google.android.tts');
      }
    } catch (e) {
      debugPrint('TTS setEngine error: $e');
    }
    await _flutterTts.setLanguage('zh-TW');
    await _flutterTts.setSpeechRate(0.5);
    await _flutterTts.setVolume(1.0);
    await _flutterTts.setPitch(1.0);
    _flutterTts.setErrorHandler((msg) {
      debugPrint('TTS error: $msg');
    });
  }

  Future<void> _speakText(String text) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty) return;
    try {
      await _flutterTts.stop();
      await _flutterTts.speak(trimmed);
    } catch (e) {
      debugPrint('TTS speak error: $e');
    }
  }

  void _confirmRecognizedWord() {
    final word = _recognizedSign?.trim();
    if (word == null || word.isEmpty) return;
    setState(() {
      _confirmedSignWords.add(word);
      _confirmedSignCandidates.add(_recognitionCandidates);
      _recognizedSign = null;
      _recognitionMessage = null;
    });
  }

  void _removeConfirmedWord(int index) {
    setState(() {
      _confirmedSignWords.removeAt(index);
      if (index < _confirmedSignCandidates.length) {
        _confirmedSignCandidates.removeAt(index);
      }
      _recognitionController.text = _confirmedSignWords.join(' ');
    });
  }

  Future<void> _editConfirmedWord(int index) async {
    FocusManager.instance.primaryFocus?.unfocus();
    final candidates = _confirmedSignCandidates[index];
    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (sheetContext) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                '選擇辨識結果',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              for (final candidate in candidates)
                ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: Text(candidate.label),
                  subtitle: Text(
                    '信心度 ${candidate.confidence.toStringAsFixed(1)}%',
                  ),
                  trailing: candidate.label == _confirmedSignWords[index]
                      ? const Icon(Icons.check_rounded)
                      : null,
                  onTap: () {
                    setState(
                      () => _confirmedSignWords[index] = candidate.label,
                    );
                    Navigator.pop(sheetContext);
                  },
                ),
              OutlinedButton.icon(
                onPressed: () {
                  Navigator.pop(sheetContext);
                  _removeConfirmedWord(index);
                },
                icon: const Icon(Icons.videocam_outlined),
                label: const Text('刪除並重拍'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _discardAndRetakeWord() async {
    setState(() {
      _recognizedSign = null;
      _recognitionCandidates = const [];
      _recognitionMessage = '請重新錄製這個詞';
    });
    await _toggleSignRecording();
  }

  void _suspendTextInputFocus() {
    FocusManager.instance.primaryFocus?.unfocus();
    _textInputFocusNode.unfocus();
    _textInputFocusNode.canRequestFocus = false;
  }

  void _restoreTextInputFocusCapability() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _textInputFocusNode.canRequestFocus = true;
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      if (_mode == TranslationMode.signToText && !_isDataContributionPageOpen) {
        _initializeCamera(cameraIndex: _selectedCameraIndex);
      }
      return;
    }
    if (_cameraController != null &&
        (state == AppLifecycleState.inactive ||
            state == AppLifecycleState.paused)) {
      unawaited(_disposeCamera());
    }
  }

  Future<void> _changeMode(TranslationMode mode) async {
    if (_mode == mode) return;
    if (mode == TranslationMode.signToText) {
      _clearRecognitionResults();
    }
    setState(() => _mode = mode);
    if (mode == TranslationMode.signToText) {
      await _initializeCamera();
    } else {
      await _disposeCamera();
    }
  }

  void _clearRecognitionResults() {
    _recognizedSign = null;
    _recognitionCandidates = const [];
    _recognitionMessage = null;
    _isRefiningSentence = false;
    _confirmedSignWords.clear();
    _confirmedSignCandidates.clear();
  }

  Future<void> _initializeCamera({int? cameraIndex}) async {
    if (_isCameraInitializing) return;
    setState(() {
      _isCameraInitializing = true;
      _recognitionMessage = null;
    });
    try {
      if (_cameras.isEmpty) _cameras = await availableCameras();
      if (_cameras.isEmpty) throw Exception('裝置沒有可用的相機');

      var selectedIndex = cameraIndex;
      if (selectedIndex == null) {
        final backIndex = _cameras.indexWhere(
          (camera) => camera.lensDirection == CameraLensDirection.back,
        );
        selectedIndex = backIndex >= 0 ? backIndex : 0;
      }
      _selectedCameraIndex = selectedIndex % _cameras.length;

      final previousController = _cameraController;
      _cameraController = null;
      await previousController?.dispose();
      final controller = CameraController(
        _cameras[_selectedCameraIndex],
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (!mounted || _mode != TranslationMode.signToText) {
        await controller.dispose();
        return;
      }
      setState(() => _cameraController = controller);
    } on CameraException catch (error) {
      await _cancelRealtimeSession();
      if (mounted) {
        setState(() {
          _recognitionMessage = error.code == 'CameraAccessDenied'
              ? '相機權限遭拒，請到裝置設定中允許相機權限'
              : '無法開啟相機：${error.description ?? error.code}';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _recognitionMessage = '無法開啟相機：$error');
    } finally {
      if (mounted) setState(() => _isCameraInitializing = false);
    }
  }

  Future<void> _disposeCamera() async {
    final controller = _cameraController;
    _cameraController = null;
    _isRecordingSign = false;
    if (controller?.value.isStreamingImages == true) {
      await controller?.stopImageStream();
    }
    await _cancelRealtimeSession();
    await controller?.dispose();
  }

  Future<void> _toggleSignRecording() async {
    final controller = _cameraController;
    if (controller == null ||
        !controller.value.isInitialized ||
        _isStartingSignRecording ||
        _isRecognizingSign) {
      return;
    }
    if (!_isRecordingSign) {
      _isStartingSignRecording = true;
      try {
        try {
          await _startRealtimeRecognition(controller);
          _usingRealtimeStream = true;
        } catch (error, stackTrace) {
          debugPrint('即時辨識串流啟動失敗：$error');
          debugPrintStack(stackTrace: stackTrace);
          if (controller.value.isStreamingImages) {
            await controller.stopImageStream();
          }
          await _cancelRealtimeSession();
          await Future<void>.delayed(const Duration(milliseconds: 250));
          await controller.startVideoRecording();
          _usingRealtimeStream = false;
        }
        if (!mounted) return;
        setState(() {
          _isRecordingSign = true;
          _recognizedSign = null;
          _recognitionMessage = null;
        });
      } on CameraException catch (error) {
        if (mounted) {
          setState(() => _recognitionMessage = '無法開始錄影：${error.description}');
        }
      } finally {
        _isStartingSignRecording = false;
      }
      return;
    }

    try {
      if (!mounted) return;
      if (_usingRealtimeStream) {
        setState(() {
          _isRecordingSign = false;
          _isRecognizingSign = true;
          _recognitionMessage = '正在辨識手語…';
        });
        await controller.stopImageStream();
        await _finishRealtimeRecognition();
      } else {
        final video = await controller.stopVideoRecording();
        setState(() {
          _isRecordingSign = false;
          _pendingWordTasks++;
          _recognitionMessage = null;
        });
        unawaited(_enqueueWordRecognition(video));
      }
    } on CameraException catch (error) {
      await _cancelRealtimeSession();
      if (mounted) {
        setState(() => _recognitionMessage = '無法結束錄影：${error.description}');
      }
    } catch (error) {
      await _cancelRealtimeSession();
      if (mounted) setState(() => _recognitionMessage = '辨識失敗：$error');
    } finally {
      if (mounted) setState(() => _isRecognizingSign = false);
    }
  }

  Future<void> _startRealtimeRecognition(CameraController controller) async {
    final response = await http
        .post(_apiUri('/api/v1/recognize-sign-stream/start'))
        .timeout(const Duration(seconds: 30));
    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (response.statusCode != 200 || decoded is! Map<String, dynamic>) {
      throw Exception('無法啟動即時辨識');
    }
    _realtimeSessionId = decoded['session_id'] as String?;
    if (_realtimeSessionId == null) throw Exception('即時辨識工作階段無效');
    _lastRealtimeFrameAt = null;
    await controller.startImageStream((image) {
      unawaited(_sendRealtimeFrame(image));
    });
  }

  Future<void> _sendRealtimeFrame(CameraImage image) async {
    final sessionId = _realtimeSessionId;
    if (sessionId == null ||
        image.planes.length != 3 ||
        _isSendingRealtimeFrame) {
      return;
    }
    final now = DateTime.now();
    if (_lastRealtimeFrameAt != null &&
        now.difference(_lastRealtimeFrameAt!).inMilliseconds < 80) {
      return;
    }
    _lastRealtimeFrameAt = now;
    _isSendingRealtimeFrame = true;
    try {
      final body = BytesBuilder(copy: false);
      for (final plane in image.planes) {
        body.add(plane.bytes);
      }
      final response = await http
          .post(
            _apiUri('/api/v1/recognize-sign-stream/$sessionId/frame'),
            headers: {
              'Content-Type': 'application/octet-stream',
              'X-Frame-Width': '${image.width}',
              'X-Frame-Height': '${image.height}',
              'X-Frame-Rotation': '${_cameraImageRotation()}',
              'X-Plane-Lengths': image.planes
                  .map((plane) => plane.bytes.length)
                  .join(','),
              'X-Row-Strides': image.planes
                  .map((plane) => plane.bytesPerRow)
                  .join(','),
              'X-Pixel-Strides': image.planes
                  .map((plane) => plane.bytesPerPixel ?? 1)
                  .join(','),
            },
            body: body.takeBytes(),
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception('即時影格處理失敗');
      }
    } catch (_) {
      // A later frame can still succeed after a transient network failure.
    } finally {
      _isSendingRealtimeFrame = false;
    }
  }

  int _cameraImageRotation() {
    final controller = _cameraController;
    if (controller == null) return 0;
    final deviceDegrees = switch (controller.value.deviceOrientation) {
      DeviceOrientation.portraitUp => 0,
      DeviceOrientation.landscapeLeft => 90,
      DeviceOrientation.portraitDown => 180,
      DeviceOrientation.landscapeRight => 270,
    };
    final sensor = controller.description.sensorOrientation;
    return controller.description.lensDirection == CameraLensDirection.front
        ? (sensor + deviceDegrees) % 360
        : (sensor - deviceDegrees + 360) % 360;
  }

  Future<void> _finishRealtimeRecognition() async {
    final sessionId = _realtimeSessionId;
    if (sessionId == null) throw Exception('即時辨識工作階段不存在');
    final response = await http
        .post(_apiUri('/api/v1/recognize-sign-stream/$sessionId/finish'))
        .timeout(const Duration(seconds: 30));
    _realtimeSessionId = null;
    _usingRealtimeStream = false;
    _applyRecognitionResponse(response);
  }

  Future<void> _cancelRealtimeSession() async {
    final sessionId = _realtimeSessionId;
    _realtimeSessionId = null;
    _usingRealtimeStream = false;
    if (sessionId == null) return;
    try {
      await http
          .delete(_apiUri('/api/v1/recognize-sign-stream/$sessionId'))
          .timeout(const Duration(seconds: 3));
    } catch (_) {}
  }

  Future<void> _enqueueWordRecognition(XFile video) async {
    try {
      final videoLength = await video.length();
      if (videoLength == 0) return;
      final request = http.MultipartRequest(
        'POST',
        _apiUri('/api/v1/recognize-sign?mode=word'),
      );
      request.files.add(
        await http.MultipartFile.fromPath(
          'video',
          video.path,
          filename: video.name.isEmpty ? 'sign_recording.mp4' : video.name,
        ),
      );
      final streamedResponse = await request.send().timeout(
        const Duration(minutes: 2),
      );
      final response = await http.Response.fromStream(streamedResponse);
      if (!mounted) return;
      if (response.statusCode == 200) {
        final decoded = jsonDecode(utf8.decode(response.bodyBytes));
        if (decoded is Map<String, dynamic>) {
          final candidates =
              (decoded['candidates'] as List<dynamic>? ?? const [])
                  .whereType<Map<String, dynamic>>()
                  .map(
                    (entry) => RecognitionCandidate(
                      label: entry['label'].toString(),
                      confidence:
                          (entry['confidence_percent'] as num?)?.toDouble() ??
                          0,
                    ),
                  )
                  .toList(growable: false);
          final label =
              decoded['label'] as String? ??
              (candidates.isEmpty ? null : candidates.first.label);
          if (label != null && label.trim().isNotEmpty) {
            final word = label.trim();
            setState(() {
              _confirmedSignWords.add(word);
              _confirmedSignCandidates.add(
                candidates.isEmpty
                    ? [RecognitionCandidate(label: word, confidence: 0)]
                    : candidates,
              );
              _recognitionController.text = _confirmedSignWords.join(' ');
              _recognitionMessage = null;
            });
            if (_autoPlaySpeech) {
              unawaited(_speakText(word));
            }
          }
        }
      }
    } catch (e) {
      debugPrint('非同步單字辨識失敗: $e');
    } finally {
      try {
        await File(video.path).delete();
      } catch (_) {}
      if (mounted) {
        setState(() {
          if (_pendingWordTasks > 0) _pendingWordTasks--;
        });
      }
    }
  }

  void _applyRecognitionResponse(http.Response response) {
    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (response.statusCode != 200) {
      final detail = decoded is Map<String, dynamic> ? decoded['detail'] : null;
      throw Exception(detail ?? '伺服器錯誤 ${response.statusCode}');
    }
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('伺服器回傳格式錯誤');
    }

    final candidates = (decoded['candidates'] as List<dynamic>? ?? const [])
        .whereType<Map<String, dynamic>>()
        .map(
          (entry) => RecognitionCandidate(
            label: entry['label'].toString(),
            confidence: (entry['confidence_percent'] as num?)?.toDouble() ?? 0,
          ),
        )
        .toList(growable: false);
    final recognizedLabel =
        decoded['label'] as String? ??
        (candidates.isEmpty ? null : candidates.first.label);
    if (!mounted) return;
    setState(() {
      _recognitionCandidates = candidates;
      _recognizedSign = null;
      if (recognizedLabel != null && recognizedLabel.trim().isNotEmpty) {
        _confirmedSignWords.add(recognizedLabel.trim());
        _confirmedSignCandidates.add(
          candidates.isEmpty
              ? [RecognitionCandidate(label: recognizedLabel, confidence: 0)]
              : candidates,
        );
      }
      _recognitionMessage = recognizedLabel != null
          ? null
          : (decoded['message'] as String? ?? '無法辨識，請重新錄製');
    });
    if (_autoPlaySpeech && recognizedLabel != null) {
      unawaited(_speakText(recognizedLabel));
    }
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2 || _isRecordingSign || _isRecognizingSign) return;
    await _initializeCamera(
      cameraIndex: (_selectedCameraIndex + 1) % _cameras.length,
    );
  }

  Future<void> _openDataContributionPage() async {
    if (_isDataContributionPageOpen) return;
    _isDataContributionPageOpen = true;
    await _disposeCamera();
    if (!mounted) {
      _isDataContributionPageOpen = false;
      return;
    }
    await Navigator.of(context).push<void>(
      MaterialPageRoute(builder: (_) => const DataContributionPage()),
    );
    _isDataContributionPageOpen = false;
    if (mounted && _mode == TranslationMode.signToText) {
      await _initializeCamera(cameraIndex: _selectedCameraIndex);
    }
  }

  Future<void> _openChatPage() async {
    if (_mode == TranslationMode.signToText) await _disposeCamera();
    if (!mounted) return;
    final workspace = await Navigator.of(context).push<ChatWorkspaceConfig>(
      MaterialPageRoute(
        builder: (_) => ChatPage(apiBaseUrl: _normalizedApiBaseUrl),
      ),
    );
    if (!mounted) return;
    if (workspace == null) {
      if (_mode == TranslationMode.signToText) {
        await _initializeCamera(cameraIndex: _selectedCameraIndex);
      }
      return;
    }
    final targetMode = workspace.mode == ChatWorkspaceMode.textToSign
        ? TranslationMode.textToSign
        : TranslationMode.signToText;
    setState(() {
      _chatWorkspace = workspace;
      _chatWorkspaceVisible = true;
      _chatSheetExtent = 0;
      _lastObservedChatMessageCount = ChatSession.instance.messages.length;
      _mode = targetMode;
      if (targetMode == TranslationMode.signToText) {
        _clearRecognitionResults();
      }
    });
    ChatSession.instance.setChatVisible(true);
    if (targetMode == TranslationMode.signToText) {
      await _initializeCamera(cameraIndex: _selectedCameraIndex);
    }
  }

  void _handleChatButton() {
    unawaited(_openChatPage());
  }

  Future<void> _sendRecognitionToChat() async {
    final text = [
      ..._confirmedSignWords,
      _recognizedSign ?? '',
    ].where((word) => word.trim().isNotEmpty).join(' ').trim();
    if (text.isEmpty) return;
    if (!ChatSession.instance.isActive) {
      unawaited(_openChatPage());
      return;
    }
    setState(() {
      _isRefiningSentence = true;
      _recognitionMessage = 'Gemma AI 正在調整中文語序…';
    });
    try {
      final response = await http
          .post(
            _apiUri('/api/v1/gloss-to-sentence'),
            headers: const {'Content-Type': 'application/json; charset=utf-8'},
            body: jsonEncode({
              'words': [
                ..._confirmedSignWords,
                if ((_recognizedSign ?? '').trim().isNotEmpty)
                  _recognizedSign!.trim(),
              ],
            }),
          )
          .timeout(const Duration(minutes: 5));
      if (response.statusCode != 200) {
        throw Exception('伺服器錯誤 ${response.statusCode}');
      }
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      final reconstructedSentence = data['sentence']?.toString().trim() ?? '';
      if (reconstructedSentence.isEmpty) {
        throw const FormatException('Gemma 沒有回傳重組後的句子');
      }
      ChatSession.instance.send(
        text: reconstructedSentence,
        source: 'sign_recognition',
      );
      if (!mounted) return;
      setState(() {
        _confirmedSignWords.clear();
        _confirmedSignCandidates.clear();
        _recognizedSign = null;
        _isRefiningSentence = false;
        _recognitionMessage = null;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _isRefiningSentence = false;
        _recognitionMessage = '無法傳送：$error';
      });
    }
  }

  Future<void> _translateToTsl(String text, {bool share = true}) async {
    final trimmedText = text.trim();
    if (trimmedText.isEmpty) {
      setState(() {
        _translationError = '請先輸入要翻譯的中文句子';
      });
      return;
    }

    FocusScope.of(context).unfocus();

    setState(() {
      _isTranslating = true;
      _translationError = null;
    });

    try {
      final response = await http
          .post(
            _apiUri('/api/v1/translate-to-tsl'),
            headers: const {'Content-Type': 'application/json; charset=utf-8'},
            body: jsonEncode({'text': trimmedText}),
          )
          .timeout(const Duration(minutes: 5));

      if (response.statusCode != 200) {
        throw Exception('伺服器錯誤 ${response.statusCode}: ${response.body}');
      }

      final data = jsonDecode(utf8.decode(response.bodyBytes));
      final tsl = data['tsl'] as String?;
      final glbPath = data['glb_url'] as String?;
      final remoteGlbUrl = glbPath == null ? null : _resolveBackendUrl(glbPath);
      final localGlbUrl = remoteGlbUrl == null
          ? _modelViewerUrl
          : await _downloadModelForViewer(remoteGlbUrl);
      final animationWords =
          (data['animation_words'] as List<dynamic>? ?? const [])
              .map((word) => word.toString())
              .toList(growable: false);
      final animationWordTimings =
          (data['animation_timeline'] as List<dynamic>? ?? const [])
              .whereType<Map<String, dynamic>>()
              .map(
                (entry) => AnimationWordTiming(
                  word: entry['word'].toString(),
                  start: (entry['start'] as num).toDouble(),
                  end: (entry['end'] as num).toDouble(),
                ),
              )
              .toList(growable: false);
      if (tsl == null || tsl.trim().isEmpty) {
        throw const FormatException('伺服器沒有回傳 TSL 結果');
      }
      if (!mounted) return;

      setState(() {
        _tslResult = tsl.trim();
        _glbUrl = remoteGlbUrl;
        _modelViewerUrl = localGlbUrl;
        _modelLoadError = null;
        _animationWords = animationWords;
        _animationWordTimings = animationWordTimings;
        _activeAnimationWordIndex = null;
        _animationPlaybackState = glbPath == null
            ? AnimationPlaybackState.idle
            : AnimationPlaybackState.playing;
        _records.insert(
          0,
          TranslationRecord(source: trimmedText, tsl: tsl.trim()),
        );
        if (_records.length > 5) _records.removeLast();
      });
      if (share && ChatSession.instance.isActive) {
        ChatSession.instance.send(
          text: trimmedText,
          source: 'text_to_sign',
          tsl: tsl.trim(),
          animationUrl: glbPath ?? '',
        );
      }
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _expandInputSheetForResult();
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _translationError = '翻譯失敗：$error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isTranslating = false;
        });
      }
    }
  }

  Future<bool> _initializeSpeech() async {
    if (_speechInitialized) return true;

    final available = await _speechToText.initialize(
      onStatus: (status) {
        if (!mounted) return;
        setState(() => _isListening = status == 'listening');
      },
      onError: (error) {
        if (!mounted) return;
        setState(() {
          _isListening = false;
          _translationError = '語音辨識失敗：${error.errorMsg}';
        });
      },
    );

    if (!available) {
      if (mounted) {
        setState(() {
          _translationError = '無法使用語音辨識，請確認已允許麥克風權限';
        });
      }
      return false;
    }

    final locales = await _speechToText.locales();
    for (final locale in locales) {
      final normalized = locale.localeId.replaceAll('-', '_').toLowerCase();
      if (normalized == 'zh_tw') {
        _taiwanChineseLocaleId = locale.localeId;
        break;
      }
    }

    _speechInitialized = true;
    return true;
  }

  Future<void> _toggleSpeechInput() async {
    if (_isListening || _speechToText.isListening) {
      await _speechToText.stop();
      if (mounted) setState(() => _isListening = false);
      return;
    }

    if (!await _initializeSpeech()) return;
    if (_taiwanChineseLocaleId == null) {
      if (mounted) {
        setState(() {
          _translationError = '手機尚未提供台灣中文語音辨識，請在系統語言設定中安裝繁體中文（台灣）';
        });
      }
      return;
    }

    setState(() {
      _translationError = null;
      _isListening = true;
    });

    await _speechToText.listen(
      onResult: (result) {
        if (!mounted) return;
        final words = result.recognizedWords;
        setState(() {
          if (words.trim().isNotEmpty) _hasEnteredSentence = true;
          _controller.text = words;
          _controller.selection = TextSelection.collapsed(offset: words.length);
        });
      },
      listenOptions: SpeechListenOptions(
        localeId: _taiwanChineseLocaleId,
        listenMode: ListenMode.dictation,
        partialResults: true,
        cancelOnError: true,
        listenFor: const Duration(minutes: 1),
        pauseFor: const Duration(seconds: 3),
      ),
    );
  }

  Future<void> _toggleChatSpeechInput() async {
    if (_isListening || _speechToText.isListening) {
      await _speechToText.stop();
      if (mounted) setState(() => _isListening = false);
      return;
    }
    if (!await _initializeSpeech()) return;
    if (_taiwanChineseLocaleId == null) {
      if (mounted) setState(() => _translationError = '手機未提供台灣中文語音辨識');
      return;
    }
    setState(() {
      _translationError = null;
      _isListening = true;
    });
    await _speechToText.listen(
      onResult: (result) {
        if (!mounted) return;
        final words = result.recognizedWords;
        setState(() {
          _chatTextController.text = words;
          _chatTextController.selection = TextSelection.collapsed(
            offset: words.length,
          );
        });
      },
      listenOptions: SpeechListenOptions(
        localeId: _taiwanChineseLocaleId,
        listenMode: ListenMode.dictation,
        partialResults: true,
        cancelOnError: true,
        listenFor: const Duration(minutes: 1),
        pauseFor: const Duration(seconds: 3),
      ),
    );
  }

  Future<void> _sendChatText() async {
    final text = _chatTextController.text.trim();
    if (text.isEmpty || _isTranslating) return;
    await _translateToTsl(text);
    if (mounted && _translationError == null) _chatTextController.clear();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    ChatSession.instance.removeListener(_handleChatSessionChanged);
    _cameraController?.dispose();
    _flutterTts.stop();
    _speechToText.cancel();
    _controller.dispose();
    _recognitionController.dispose();
    _chatTextController.dispose();
    _textInputFocusNode.dispose();
    _inputSheetController.dispose();
    _signInputSheetController.dispose();
    _chatMessagesController.dispose();
    for (final file in _cachedModelFiles) {
      unawaited(file.delete().catchError((_) => file));
    }
    super.dispose();
  }

  void _togglePlaybackSpeed() {
    setState(() {
      _playbackSpeed = switch (_playbackSpeed) {
        0.5 => 1,
        1 => 1.5,
        _ => 0.5,
      };
      _viewerRevision++;
    });
  }

  void _replayAvatarAnimation() {
    setState(() => _viewerRevision++);
  }

  Future<void> _toggleAnimationPlayback() async {
    if (_glbUrl == null) {
      _replayAvatarAnimation();
      return;
    }
    final controller = _modelViewerController;
    if (_animationPlaybackState == AnimationPlaybackState.playing) {
      await controller?.runJavaScript(
        "document.querySelector('model-viewer')?.pause();",
      );
      if (mounted) {
        setState(() => _animationPlaybackState = AnimationPlaybackState.paused);
      }
      return;
    }
    if (_animationPlaybackState == AnimationPlaybackState.paused) {
      await controller?.runJavaScript(
        "document.querySelector('model-viewer')?.play({repetitions: 1});",
      );
      if (mounted) {
        setState(
          () => _animationPlaybackState = AnimationPlaybackState.playing,
        );
      }
      return;
    }
    _replayAvatarAnimation();
  }

  void _handleAnimationMessage(JavaScriptMessage message) {
    final data = jsonDecode(message.message) as Map<String, dynamic>;
    if (!mounted) return;
    setState(() {
      switch (data['type']) {
        case 'playing':
          if (_glbUrl != null) {
            _animationPlaybackState = AnimationPlaybackState.playing;
          }
        case 'paused':
          if (_glbUrl != null &&
              _animationPlaybackState != AnimationPlaybackState.completed) {
            _animationPlaybackState = AnimationPlaybackState.paused;
          }
        case 'completed':
          _animationPlaybackState = AnimationPlaybackState.completed;
          _activeAnimationWordIndex = null;
        case 'word':
          final index = data['index'] as int?;
          if (index == -1) {
            _activeAnimationWordIndex = null;
          } else if (index != null &&
              index >= 0 &&
              index < _animationWords.length) {
            _activeAnimationWordIndex = index;
          }
      }
    });
  }

  void _resizeInputSheet(DragUpdateDetails details) {
    if (!_inputSheetController.isAttached) return;
    final viewportHeight = MediaQuery.sizeOf(context).height;
    final minimumExtent = _tslResult == null
        ? _minInputSheetExtent
        : _resultMinInputSheetExtent;
    final nextExtent = (_inputSheetExtent - details.delta.dy / viewportHeight)
        .clamp(minimumExtent, _maxInputSheetExtent);
    _inputSheetController.jumpTo(nextExtent);
  }

  void _expandInputSheetForResult() {
    if (!_inputSheetController.isAttached ||
        _inputSheetExtent >= _resultInputSheetExtent) {
      return;
    }
    unawaited(
      _inputSheetController.animateTo(
        _resultInputSheetExtent,
        duration: const Duration(milliseconds: 280),
        curve: Curves.easeOutCubic,
      ),
    );
  }

  Widget _buildFixedChatButton() {
    final session = ChatSession.instance;

    return Positioned(
      key: const Key('fixed-chat-button'),
      left: 12,
      top: 12,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          _RoundOverlayButton(
            tooltip: '聊天室',
            icon: session.isActive ? Icons.forum_rounded : Icons.forum_outlined,
            onPressed: _handleChatButton,
          ),
          if (session.unreadCount > 0)
            Positioned(
              left: -3,
              top: -5,
              child: Container(
                constraints: const BoxConstraints(minWidth: 24, minHeight: 24),
                padding: const EdgeInsets.symmetric(horizontal: 6),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: const Color(0xFF1689F5),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.white, width: 2),
                ),
                child: Text(
                  session.unreadCount > 99 ? '99+' : '${session.unreadCount}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _leaveChat() async {
    await ChatSession.instance.leave();
    if (!mounted) return;
    await _disposeCamera();
    if (!mounted) return;
    setState(() {
      _chatWorkspace = null;
      _chatWorkspaceVisible = false;
      _chatSheetExtent = 0;
      _lastObservedChatMessageCount = 0;
      _inputSheetExtent = 0.22;
      _mode = TranslationMode.textToSign;
      _controller.clear();
      _chatTextController.clear();
      _tslResult = null;
      _translationError = null;
      _glbUrl = null;
      _animationWords = const [];
      _animationWordTimings = const [];
      _activeAnimationWordIndex = null;
      _animationPlaybackState = AnimationPlaybackState.idle;
      _clearRecognitionResults();
    });
  }

  Widget _buildLeaveChatButton() {
    return Positioned(
      left: 12,
      top: 12,
      child: _RoundOverlayButton(
        key: const Key('leave-chat-button'),
        tooltip: '結束對話',
        icon: Icons.logout_rounded,
        onPressed: () => unawaited(_leaveChat()),
      ),
    );
  }

  Widget _buildChatMessages(ScrollController scrollController) {
    final session = ChatSession.instance;
    if (session.messages.isEmpty) {
      return const Center(
        child: Text('尚無訊息', style: TextStyle(color: Colors.black54)),
      );
    }
    return ListView.builder(
      controller: scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      itemCount: session.messages.length,
      itemBuilder: (context, index) {
        final message = session.messages[index];
        final mine = session.localMode
            ? message.source != 'sign_recognition'
            : message.senderId == session.deviceId;
        final hasAnimation =
            message.tsl.isNotEmpty || message.animationUrl.isNotEmpty;
        return Align(
          alignment: mine ? Alignment.centerRight : Alignment.centerLeft,
          child: Container(
            constraints: const BoxConstraints(maxWidth: 330),
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.fromLTRB(13, 9, 13, 8),
            decoration: BoxDecoration(
              color: mine ? const Color(0xFF4D8DFF) : Colors.white,
              borderRadius: BorderRadius.circular(17),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  message.text,
                  style: TextStyle(
                    color: mine ? Colors.white : Colors.black87,
                    fontSize: 16,
                  ),
                ),
                if (message.tsl.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Text(
                    message.tsl,
                    style: TextStyle(
                      color: mine ? Colors.white70 : Colors.black54,
                      fontSize: 12,
                    ),
                  ),
                ],
                if (hasAnimation) ...[
                  const SizedBox(height: 6),
                  ChatSignThumbnail(
                    apiBaseUrl: _normalizedApiBaseUrl,
                    message: message,
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildChatWorkspaceSheet(double availableHeight) {
    final isTextMode = _chatWorkspace!.mode == ChatWorkspaceMode.textToSign;
    const fixedControlsHeight = 120.0;
    const topControlsClearance = 68.0;
    final maximumExtent =
        ((availableHeight - topControlsClearance) / availableHeight).clamp(
          0.50,
          0.92,
        );
    final minimumExtent = (fixedControlsHeight / availableHeight).clamp(
      0.12,
      maximumExtent,
    );
    final effectiveExtent = _chatSheetExtent.clamp(
      minimumExtent,
      maximumExtent,
    );
    final currentHeight = effectiveExtent * availableHeight;
    final showHeader = currentHeight >= fixedControlsHeight + 42;
    final showError = currentHeight >= fixedControlsHeight + 70;
    return Positioned.fill(
      key: const Key('chat-workspace-sheet'),
      child: Align(
        alignment: Alignment.bottomCenter,
        child: FractionallySizedBox(
          widthFactor: 1,
          heightFactor: effectiveExtent,
          child: Material(
            key: const Key('chat-sheet-surface'),
            elevation: 12,
            color: const Color(0xFFE4E4E4),
            borderRadius: BorderRadius.vertical(top: const Radius.circular(26)),
            clipBehavior: Clip.antiAlias,
            child: Column(
              children: [
                GestureDetector(
                  key: const Key('chat-sheet-drag-handle'),
                  behavior: HitTestBehavior.opaque,
                  onVerticalDragUpdate: (details) {
                    final next =
                        (effectiveExtent - details.delta.dy / availableHeight)
                            .clamp(minimumExtent, maximumExtent);
                    setState(() => _chatSheetExtent = next);
                    ChatSession.instance.setChatVisible(
                      next > minimumExtent + 0.025,
                    );
                  },
                  child: SizedBox(
                    height: 32,
                    child: Center(
                      child: Container(
                        width: 48,
                        height: 5,
                        decoration: BoxDecoration(
                          color: Colors.black26,
                          borderRadius: BorderRadius.circular(3),
                        ),
                      ),
                    ),
                  ),
                ),
                if (showHeader)
                  SizedBox(
                    height: 42,
                    child: Row(
                      children: [
                        const SizedBox(width: 14),
                        Expanded(
                          child: const Text(
                            '聊天室',
                            style: TextStyle(fontWeight: FontWeight.w700),
                          ),
                        ),
                      ],
                    ),
                  ),
                Expanded(child: _buildChatMessages(_chatMessagesController)),
                if (isTextMode)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(12, 6, 12, 12),
                    child: TextField(
                      controller: _chatTextController,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendChatText(),
                      decoration: InputDecoration(
                        hintText: '輸入要翻譯的中文…',
                        filled: true,
                        fillColor: Colors.white,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24),
                          borderSide: BorderSide.none,
                        ),
                        suffixIcon: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                              tooltip: _isListening ? '停止語音輸入' : '語音輸入',
                              onPressed: _isTranslating
                                  ? null
                                  : _toggleChatSpeechInput,
                              icon: Icon(
                                _isListening
                                    ? Icons.stop_circle_rounded
                                    : Icons.mic_rounded,
                              ),
                            ),
                            IconButton(
                              tooltip: '翻譯並傳送',
                              onPressed: _isTranslating ? null : _sendChatText,
                              icon: const Icon(Icons.send_rounded),
                            ),
                            const SizedBox(width: 4),
                          ],
                        ),
                      ),
                    ),
                  )
                else
                  Padding(
                    padding: const EdgeInsets.fromLTRB(12, 2, 12, 8),
                    child: _buildCameraControls(),
                  ),
                if (showError && _translationError != null && isTextMode)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Text(
                      _translationError!,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.error,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _viewerGray,
      body: SafeArea(
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 220),
          child: _mode == TranslationMode.textToSign
              ? _buildTextToSignPage()
              : _buildSignToTextLayout(),
        ),
      ),
    );
  }

  Widget _buildTextToSignPage() {
    return LayoutBuilder(
      key: const ValueKey('text-to-sign'),
      builder: (context, constraints) {
        final minimumExtent = _tslResult == null
            ? _minInputSheetExtent
            : _resultMinInputSheetExtent;
        final sheetTop = constraints.maxHeight * (1 - _inputSheetExtent);
        return Stack(
          fit: StackFit.expand,
          children: [
            _buildAvatarViewer(),
            Positioned(right: 12, top: 12, child: _buildHomeMenu()),
            Positioned(
              right: 10,
              bottom: constraints.maxHeight - sheetTop + 52,
              child: Column(
                children: [
                  _RoundOverlayButton(
                    key: const Key('playback-speed-button'),
                    tooltip: '調整播放速度',
                    label: '${_playbackSpeed.toStringAsFixed(1)}x',
                    onPressed: _togglePlaybackSpeed,
                  ),
                  const SizedBox(height: 8),
                  _RoundOverlayButton(
                    key: const Key('replay-button'),
                    tooltip:
                        _glbUrl != null &&
                            _animationPlaybackState ==
                                AnimationPlaybackState.playing
                        ? '暫停動畫'
                        : _glbUrl != null &&
                              _animationPlaybackState ==
                                  AnimationPlaybackState.paused
                        ? '繼續播放'
                        : '重新播放',
                    icon:
                        _glbUrl != null &&
                            _animationPlaybackState ==
                                AnimationPlaybackState.playing
                        ? Icons.pause_rounded
                        : _glbUrl != null &&
                              _animationPlaybackState ==
                                  AnimationPlaybackState.paused
                        ? Icons.play_arrow_rounded
                        : Icons.replay_rounded,
                    onPressed: _toggleAnimationPlayback,
                  ),
                ],
              ),
            ),
            if (!_chatWorkspaceVisible)
              NotificationListener<DraggableScrollableNotification>(
                onNotification: (notification) {
                  if ((_inputSheetExtent - notification.extent).abs() > 0.002) {
                    setState(() => _inputSheetExtent = notification.extent);
                  }
                  return false;
                },
                child: DraggableScrollableSheet(
                  controller: _inputSheetController,
                  initialChildSize: _minInputSheetExtent,
                  minChildSize: minimumExtent,
                  maxChildSize: _maxInputSheetExtent,
                  snap: true,
                  snapSizes: [minimumExtent, _maxInputSheetExtent],
                  builder: (context, scrollController) =>
                      _buildInputSheet(scrollController),
                ),
              ),
            if (_chatWorkspaceVisible && _chatWorkspace != null)
              _buildChatWorkspaceSheet(constraints.maxHeight),
            if (_chatWorkspaceVisible && _chatWorkspace != null)
              _buildLeaveChatButton()
            else
              _buildFixedChatButton(),
          ],
        );
      },
    );
  }

  Widget _buildSignToTextLayout() {
    return LayoutBuilder(
      key: const ValueKey('sign-to-text-layout'),
      builder: (context, constraints) => Stack(
        fit: StackFit.expand,
        children: [
          ColoredBox(color: _viewerGray, child: _buildCameraPreview()),
          Positioned(right: 12, top: 12, child: _buildHomeMenu()),
          if (_isRecordingSign)
            const Positioned(
              left: 72,
              right: 72,
              top: 12,
              height: 52,
              child: Center(
                child: Chip(
                  avatar: Icon(
                    Icons.fiber_manual_record,
                    color: Colors.red,
                    size: 18,
                  ),
                  label: Text('錄影中'),
                ),
              ),
            ),
          if (_confirmedSignWords.isNotEmpty || _recognizedSign != null)
            Positioned(
              left: 18,
              right: 18,
              top: 74,
              child: _buildRecognitionComposer(),
            ),
          if (_recognitionMessage != null &&
              _confirmedSignWords.isEmpty &&
              _recognizedSign == null &&
              !_isRecordingSign)
            Positioned(
              left: 24,
              right: 24,
              top: 82,
              child: Text(
                _recognitionMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white),
              ),
            ),
          if (!_chatWorkspaceVisible)
            Positioned(
              left: 24,
              right: 24,
              bottom: 72,
              child: _buildCameraControls(),
            ),
          if (!_chatWorkspaceVisible)
            Positioned(
              left: 12,
              right: 12,
              bottom: 10,
              child: _ModeSwitch(mode: _mode, onChanged: _changeMode),
            ),
          if (_chatWorkspaceVisible && _chatWorkspace != null)
            _buildChatWorkspaceSheet(constraints.maxHeight),
          if (_chatWorkspaceVisible && _chatWorkspace != null)
            _buildLeaveChatButton()
          else
            _buildFixedChatButton(),
        ],
      ),
    );
  }

  Widget _buildRecognitionComposer() {
    return Material(
      color: Colors.white.withValues(alpha: 0.92),
      borderRadius: BorderRadius.circular(20),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_confirmedSignWords.isNotEmpty) ...[
              const Text(
                '目前句子',
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  for (
                    var index = 0;
                    index < _confirmedSignWords.length;
                    index++
                  )
                    InputChip(
                      label: Text(_confirmedSignWords[index]),
                      onPressed: () => _editConfirmedWord(index),
                      onDeleted: () => _removeConfirmedWord(index),
                      deleteIcon: const Icon(Icons.close, size: 17),
                      backgroundColor: Colors.white,
                      side: const BorderSide(color: Colors.black12),
                      visualDensity: VisualDensity.compact,
                    ),
                ],
              ),
            ],
            if (_confirmedSignWords.isNotEmpty && _recognizedSign != null)
              const Divider(height: 22),
            if (_recognizedSign != null) ...[
              const Text(
                '這個詞辨識為',
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
              const SizedBox(height: 5),
              Row(
                children: [
                  Expanded(
                    child: Text(
                      _recognizedSign!,
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  TextButton.icon(
                    key: const Key('retake-word-button'),
                    onPressed: _discardAndRetakeWord,
                    icon: const Icon(Icons.replay_rounded, size: 19),
                    label: const Text('重拍'),
                  ),
                  FilledButton.icon(
                    key: const Key('confirm-word-button'),
                    onPressed: _confirmRecognizedWord,
                    icon: const Icon(Icons.check_rounded, size: 19),
                    label: const Text('加入'),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildHomeMenu() {
    return PopupMenuButton<String>(
      key: const Key('home-menu-button'),
      tooltip: '開啟選單',
      color: Colors.white,
      surfaceTintColor: Colors.transparent,
      elevation: 5,
      offset: const Offset(0, 48),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      onOpened: _suspendTextInputFocus,
      onCanceled: _restoreTextInputFocusCapability,
      onSelected: (value) {
        if (value == 'toggle-animation') {
          setState(() {
            _autoPlayAnimation = !_autoPlayAnimation;
            _viewerRevision++;
          });
        } else if (value == 'toggle-speech') {
          setState(() => _autoPlaySpeech = !_autoPlaySpeech);
        } else if (value == 'contribute') {
          _openDataContributionPage();
        }
        _restoreTextInputFocusCapability();
      },
      itemBuilder: (context) => [
        _toggleMenuItem(
          value: 'toggle-animation',
          label: '自動播放動畫',
          enabled: _autoPlayAnimation,
        ),
        _toggleMenuItem(
          value: 'toggle-speech',
          label: '自動播放語音',
          enabled: _autoPlaySpeech,
        ),
        const PopupMenuItem<String>(
          value: 'contribute',
          child: Text('協助增加資料庫'),
        ),
      ],
      child: const _RoundOverlayButton(
        tooltip: '開啟選單',
        icon: Icons.menu_rounded,
      ),
    );
  }

  PopupMenuItem<String> _toggleMenuItem({
    required String value,
    required String label,
    required bool enabled,
  }) {
    return PopupMenuItem<String>(
      value: value,
      child: Row(
        children: [
          Expanded(child: Text(label)),
          IgnorePointer(
            child: Switch.adaptive(value: enabled, onChanged: (_) {}),
          ),
        ],
      ),
    );
  }

  Widget _buildInputSheet(ScrollController scrollController) {
    final useExpandedInputLayout =
        _inputSheetExtent >= _expandedInputSheetThreshold;
    return Column(
      children: [
        Expanded(
          child: Material(
            color: const Color(0xFFF1F1F1),
            elevation: 8,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 7, 14, 10),
              child: Column(
                children: [
                  GestureDetector(
                    key: const Key('input-sheet-drag-handle'),
                    behavior: HitTestBehavior.opaque,
                    onVerticalDragUpdate: _resizeInputSheet,
                    child: SizedBox(
                      width: 96,
                      height: 16,
                      child: Center(
                        child: Container(
                          width: 42,
                          height: 5,
                          decoration: BoxDecoration(
                            color: Colors.black26,
                            borderRadius: BorderRadius.circular(3),
                          ),
                        ),
                      ),
                    ),
                  ),
                  Expanded(
                    child: _buildTextInput(
                      scrollController,
                      isExpanded: useExpandedInputLayout,
                    ),
                  ),
                  if (_translationError != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 5),
                      child: Text(
                        _translationError!,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.error,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  if (MediaQuery.viewInsetsOf(context).bottom == 0) ...[
                    const SizedBox(height: 4),
                    _ModeSwitch(
                      mode: _mode,
                      onChanged: _changeMode,
                      borderless: true,
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildAvatarViewer() {
    final modelUrl = _modelViewerUrl;
    final shouldPlaySentence = _glbUrl != null;
    final encodedAnimationTimeline = jsonEncode(
      _animationWordTimings
          .map((timing) => {'start': timing.start, 'end': timing.end})
          .toList(growable: false),
    );

    return Stack(
      fit: StackFit.expand,
      children: [
        ColoredBox(
          color: _viewerGray,
          child: !widget.enableModelViewer
              ? const ColoredBox(color: _viewerGray)
              : modelUrl != null
              ? ModelViewer(
                  key: ValueKey(
                    '$modelUrl-$_viewerRevision-$_hasEnteredSentence',
                  ),
                  src: modelUrl,
                  alt: '台灣手語動畫',
                  backgroundColor: _viewerGray,
                  // idle.glb 的自然站姿儲存在動畫軌，而不是模型的綁定姿勢。
                  // 尚未翻譯時自動播放 idle，避免停在雙臂展開的第 0 幀。
                  // idle 必須持續套用動畫軌，否則模型會回到雙臂水平張開的綁定姿勢。
                  // 「自動播放動畫」只控制句子動畫是否重複播放。
                  autoPlay: !shouldPlaySentence,
                  autoRotate: false,
                  cameraControls: true,
                  cameraOrbit: _cameraOrbit,
                  cameraTarget: _cameraTarget,
                  minCameraOrbit: 'auto auto 25%',
                  interactionPrompt: _hasEnteredSentence
                      ? InteractionPrompt.none
                      : InteractionPrompt.auto,
                  loading: Loading.eager,
                  debugLogging: false,
                  onWebViewCreated: (controller) {
                    _modelViewerController = controller;
                    unawaited(
                      controller.addJavaScriptChannel(
                        'AnimationState',
                        onMessageReceived: _handleAnimationMessage,
                      ),
                    );
                  },
                  relatedJs:
                      '''
const viewer = document.querySelector('model-viewer');
const sendAnimationState = (payload) => {
  if (window.AnimationState) {
    window.AnimationState.postMessage(JSON.stringify(payload));
  }
};
let lastWordIndex = -1;
const wordTimeline = $encodedAnimationTimeline;
const updateActiveWord = () => {
  const index = wordTimeline.findIndex(
    ({start, end}) => viewer.currentTime >= start && viewer.currentTime < end
  );
  if (index !== lastWordIndex) {
    lastWordIndex = index;
    sendAnimationState({type: 'word', index});
  }
  if (!viewer.paused) requestAnimationFrame(updateActiveWord);
};
viewer.addEventListener('play', () => {
  sendAnimationState({type: 'playing'});
  requestAnimationFrame(updateActiveWord);
});
viewer.addEventListener('pause', () => {
  sendAnimationState({type: 'paused'});
});
viewer.addEventListener('finished', () => {
  sendAnimationState({type: 'completed'});
});
viewer.addEventListener('load', () => {
  viewer.timeScale = $_playbackSpeed;
  ${shouldPlaySentence ? "viewer.play({repetitions: ${_autoPlayAnimation ? 'Infinity' : '1'}});" : ''}
}, {once: true});
''',
                )
              : Center(
                  child: _modelLoadError == null
                      ? const CircularProgressIndicator()
                      : Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(Icons.error_outline_rounded, size: 32),
                            const SizedBox(height: 8),
                            const Text('Unable to load 3D model'),
                            const SizedBox(height: 8),
                            TextButton(
                              onPressed: _loadInitialModel,
                              child: const Text('Retry'),
                            ),
                          ],
                        ),
                ),
        ),
        if (_isTranslating)
          const ColoredBox(
            color: Color(0x66000000),
            child: Center(
              child: CircularProgressIndicator(color: Colors.white),
            ),
          ),
      ],
    );
  }

  Widget _buildTextInput(
    ScrollController scrollController, {
    required bool isExpanded,
  }) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                TextField(
                  controller: _controller,
                  focusNode: _textInputFocusNode,
                  scrollController: scrollController,
                  expands: true,
                  minLines: null,
                  maxLines: null,
                  textAlignVertical: isExpanded
                      ? TextAlignVertical.top
                      : TextAlignVertical.center,
                  textInputAction: TextInputAction.send,
                  style: const TextStyle(fontSize: _inputTextSize),
                  onChanged: (value) {
                    if (!_hasEnteredSentence && value.trim().isNotEmpty) {
                      setState(() => _hasEnteredSentence = true);
                    }
                  },
                  onSubmitted: _isTranslating ? null : _translateToTsl,
                  decoration: InputDecoration(
                    hintText: '輸入文字…',
                    border: InputBorder.none,
                    contentPadding: isExpanded
                        ? const EdgeInsets.fromLTRB(18, 8, 104, 8)
                        : const EdgeInsets.fromLTRB(18, 0, 104, 0),
                  ),
                ),
                Positioned(
                  right: 5,
                  bottom: 0,
                  top: isExpanded ? null : 0,
                  child: Row(
                    children: [
                      IconButton(
                        key: const Key('speech-input-button'),
                        tooltip: _isListening ? '停止語音輸入' : '語音輸入（台灣中文）',
                        onPressed: _isTranslating ? null : _toggleSpeechInput,
                        icon: Icon(
                          _isListening
                              ? Icons.stop_circle_rounded
                              : Icons.mic_rounded,
                          color: _isListening ? Colors.red : Colors.black87,
                        ),
                      ),
                      IconButton(
                        key: const Key('text-send-button'),
                        tooltip: '傳送',
                        onPressed: _isTranslating
                            ? null
                            : () => _translateToTsl(_controller.text),
                        icon: const Icon(
                          Icons.send_rounded,
                          color: Colors.black87,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          if (_tslResult != null) ...[
            const Divider(height: 1, indent: 12, endIndent: 12),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(18, 12, 18, 10),
                child: Align(
                  alignment: Alignment.topLeft,
                  child: _buildTslResultText(),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildTslResultText() {
    final words = _tslResult!.split(RegExp(r'\s+'));
    final activeWord = _activeAnimationWordIndex == null
        ? null
        : _animationWords[_activeAnimationWordIndex!];
    return Text.rich(
      TextSpan(
        children: [
          for (var index = 0; index < words.length; index++) ...[
            TextSpan(
              text: words[index],
              style: TextStyle(
                color: words[index] == activeWord
                    ? const Color(0xFFFF7A00)
                    : Colors.black54,
                fontSize: _inputTextSize,
                fontWeight: words[index] == activeWord
                    ? FontWeight.w700
                    : FontWeight.w400,
              ),
            ),
            if (index < words.length - 1) const TextSpan(text: ' '),
          ],
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    final controller = _cameraController;
    if (_isCameraInitializing) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }
    if (controller == null || !controller.value.isInitialized) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.videocam_off_outlined,
              size: 84,
              color: Colors.white70,
            ),
            const SizedBox(height: 12),
            FilledButton(
              onPressed: _initializeCamera,
              child: const Text('開啟相機'),
            ),
          ],
        ),
      );
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            final previewSize = controller.value.previewSize;
            if (previewSize == null) return const SizedBox.expand();
            return ClipRect(
              child: FittedBox(
                fit: BoxFit.cover,
                alignment: Alignment.center,
                child: SizedBox(
                  width: previewSize.height,
                  height: previewSize.width,
                  child: CameraPreview(controller),
                ),
              ),
            );
          },
        ),
      ],
    );
  }

  Widget _buildCameraControls() {
    final isBusy = _isCameraInitializing || _isStartingSignRecording;
    return SizedBox(
      height: 68,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _RoundOverlayButton(
            tooltip: '切換鏡頭',
            icon: Icons.cameraswitch_outlined,
            onPressed: _cameras.length > 1 && !_isRecordingSign && !isBusy
                ? _switchCamera
                : null,
          ),
          InkWell(
            customBorder: const CircleBorder(),
            onTap: isBusy ? null : _toggleSignRecording,
            child: Container(
              width: 52,
              height: 52,
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.92),
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white, width: 2),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x55000000),
                    blurRadius: 8,
                    offset: Offset(0, 2),
                  ),
                ],
              ),
              alignment: Alignment.center,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 150),
                width: _isRecordingSign ? 19 : 38,
                height: _isRecordingSign ? 19 : 38,
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(
                    _isRecordingSign ? 5 : 19,
                  ),
                ),
              ),
            ),
          ),
          _RoundOverlayButton(
            key: Key('recognition-send-button'),
            tooltip: ChatSession.instance.isActive ? '傳送辨識結果' : '先開始對話',
            icon: Icons.send_rounded,
            onPressed: _confirmedSignWords.isNotEmpty || _recognizedSign != null
                ? (_isRefiningSentence ? null : _sendRecognitionToChat)
                : null,
          ),
        ],
      ),
    );
  }
}

class _ModeSwitch extends StatelessWidget {
  const _ModeSwitch({
    required this.mode,
    required this.onChanged,
    this.borderless = false,
  });

  final TranslationMode mode;
  final ValueChanged<TranslationMode> onChanged;
  final bool borderless;

  @override
  Widget build(BuildContext context) {
    final textToSign = mode == TranslationMode.textToSign;

    return Container(
      height: 52,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: borderless ? Colors.transparent : Colors.white,
        border: borderless
            ? null
            : Border.all(color: Colors.black38, width: 1.5),
        borderRadius: BorderRadius.circular(28),
      ),
      child: Row(
        children: [
          Expanded(
            child: TextButton(
              onPressed: () => onChanged(
                textToSign
                    ? TranslationMode.textToSign
                    : TranslationMode.signToText,
              ),
              child: Text(
                textToSign ? '偵測語言' : '台灣手語',
                style: const TextStyle(
                  color: Color(0xFF4D8DFF),
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          Container(
            width: 32,
            height: 32,
            decoration: const BoxDecoration(
              color: Color(0xFFDCE8FF),
              shape: BoxShape.circle,
            ),
            child: IconButton(
              padding: EdgeInsets.zero,
              onPressed: () => onChanged(
                textToSign
                    ? TranslationMode.signToText
                    : TranslationMode.textToSign,
              ),
              icon: const Icon(
                Icons.swap_horiz_rounded,
                size: 20,
                color: Color(0xFF4D8DFF),
              ),
            ),
          ),
          Expanded(
            child: TextButton(
              onPressed: () => onChanged(
                textToSign
                    ? TranslationMode.signToText
                    : TranslationMode.textToSign,
              ),
              child: Text(
                textToSign ? '台灣手語' : '繁體中文',
                style: const TextStyle(color: Colors.black45, fontSize: 16),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _RoundOverlayButton extends StatelessWidget {
  const _RoundOverlayButton({
    super.key,
    required this.tooltip,
    this.icon,
    this.label,
    this.onPressed,
  });

  final String tooltip;
  final IconData? icon;
  final String? label;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final content = Container(
      width: 44,
      height: 44,
      alignment: Alignment.center,
      decoration: const BoxDecoration(
        color: Colors.white,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: Color(0x33000000),
            blurRadius: 7,
            offset: Offset(0, 3),
          ),
        ],
      ),
      child: label != null
          ? Text(
              label!,
              style: const TextStyle(
                color: Colors.black54,
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            )
          : Icon(icon, color: Colors.black87, size: 27),
    );

    if (onPressed == null) return Tooltip(message: tooltip, child: content);
    return Tooltip(
      message: tooltip,
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onPressed,
        child: content,
      ),
    );
  }
}
