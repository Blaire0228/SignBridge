import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

class DataContributionPage extends StatefulWidget {
  const DataContributionPage({super.key});

  @override
  State<DataContributionPage> createState() => _DataContributionPageState();
}

class _DataContributionPageState extends State<DataContributionPage>
    with WidgetsBindingObserver {
  final TextEditingController _wordController = TextEditingController();

  CameraController? _cameraController;
  List<CameraDescription> _cameras = const [];
  int _selectedCameraIndex = 0;
  XFile? _recordedVideo;
  VideoPlayerController? _videoPlayerController;
  bool _isInitializing = true;
  bool _isRecording = false;
  bool _isChangingRecordingState = false;
  bool _isClosing = false;
  bool _allowPop = false;
  String? _cameraError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }

  Future<void> _initializeCamera({int? cameraIndex}) async {
    if (mounted) {
      setState(() {
        _isInitializing = true;
        _cameraError = null;
      });
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw Exception('裝置沒有可用的相機');
      _cameras = cameras;
      final backIndex = cameras.indexWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
      );
      _selectedCameraIndex = cameraIndex ?? (backIndex >= 0 ? backIndex : 0);
      final camera = cameras[_selectedCameraIndex];
      final previous = _cameraController;
      _cameraController = null;
      await previous?.dispose();
      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _cameraController = controller);
    } on CameraException catch (error) {
      if (mounted) {
        setState(() {
          _cameraError = error.code == 'CameraAccessDenied'
              ? '相機權限遭拒，請到裝置設定中允許相機權限'
              : '無法開啟相機：${error.description ?? error.code}';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _cameraError = '無法開啟相機：$error');
    } finally {
      if (mounted) setState(() => _isInitializing = false);
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    } else if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _cameraController?.dispose();
      _cameraController = null;
      _isRecording = false;
    }
  }

  Future<void> _toggleRecording() async {
    final controller = _cameraController;
    if (controller == null ||
        !controller.value.isInitialized ||
        _isChangingRecordingState) {
      return;
    }
    setState(() => _isChangingRecordingState = true);
    try {
      if (_isRecording) {
        final video = await controller.stopVideoRecording();
        final player = VideoPlayerController.file(File(video.path));
        await player.initialize();
        await player.setLooping(true);
        await _videoPlayerController?.dispose();
        if (!mounted) {
          await player.dispose();
          return;
        }
        setState(() {
          _isRecording = false;
          _recordedVideo = video;
          _videoPlayerController = player;
        });
      } else {
        final previousPlayer = _videoPlayerController;
        _videoPlayerController = null;
        if (mounted) {
          setState(() => _recordedVideo = null);
          await WidgetsBinding.instance.endOfFrame;
        }
        await previousPlayer?.dispose();
        await controller.startVideoRecording();
        if (!mounted) return;
        setState(() {
          _isRecording = true;
          _recordedVideo = null;
        });
      }
    } on CameraException catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('錄影失敗：${error.description ?? error.code}')),
        );
      }
    } finally {
      if (mounted) setState(() => _isChangingRecordingState = false);
    }
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2 || _isRecording || _isChangingRecordingState) {
      return;
    }
    await _initializeCamera(
      cameraIndex: (_selectedCameraIndex + 1) % _cameras.length,
    );
  }

  Future<void> _closePage() async {
    if (_isClosing) return;
    _isClosing = true;
    if (_isRecording) {
      try {
        await _cameraController?.stopVideoRecording();
      } on CameraException {
        // Closing the page must still succeed if recording already stopped.
      }
    }
    if (!mounted) return;
    setState(() {
      _isRecording = false;
      _allowPop = true;
    });
    // Let PopScope rebuild with canPop=true before requesting the route pop.
    await WidgetsBinding.instance.endOfFrame;
    if (mounted) Navigator.of(context).pop();
  }

  Future<void> _prepareUpload() async {
    final word = _wordController.text.trim();
    if (word.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('請輸入這個手語詞的中文')));
      return;
    }
    final recordedVideo = _recordedVideo;
    if (recordedVideo == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('請先完成一段手語錄影')));
      return;
    }
    FocusScope.of(context).unfocus();
    final player = _videoPlayerController;
    setState(() {
      _wordController.clear();
      _recordedVideo = null;
      _videoPlayerController = null;
    });
    // Remove VideoPlayer from the tree before disposing its controller.
    await WidgetsBinding.instance.endOfFrame;
    await player?.dispose();
    try {
      final file = File(recordedVideo.path);
      if (await file.exists()) await file.delete();
    } on FileSystemException {
      // The camera plugin may already have cleaned up its temporary file.
    }
    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('已上傳成功')));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _videoPlayerController?.dispose();
    _wordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: _allowPop,
      onPopInvokedWithResult: (didPop, result) {
        if (!didPop) _closePage();
      },
      child: Scaffold(
        backgroundColor: const Color(0xFFC6C6C6),
        body: SafeArea(
          child: Stack(
            fit: StackFit.expand,
            children: [
              ColoredBox(
                color: const Color(0xFFC6C6C6),
                child: _buildCameraPreview(),
              ),
              Positioned(
                left: 14,
                top: 12,
                child: _ContributionCircleButton(
                  key: const Key('contribution-back-button'),
                  tooltip: '返回',
                  icon: Icons.arrow_back_rounded,
                  onPressed: _closePage,
                ),
              ),
              Positioned(right: 14, top: 12, child: _buildMenu()),
              if (_isRecording)
                const Positioned(
                  top: 12,
                  left: 74,
                  right: 74,
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
              Positioned(
                left: 16,
                right: 16,
                bottom: 14,
                child: _buildBottomPanel(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMenu() {
    return PopupMenuButton<String>(
      tooltip: '開啟選單',
      color: Colors.white,
      surfaceTintColor: Colors.transparent,
      offset: const Offset(0, 48),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      onOpened: () => FocusManager.instance.primaryFocus?.unfocus(),
      itemBuilder: (context) => const [
        PopupMenuItem<String>(enabled: false, child: Text('自動播放動畫')),
        PopupMenuItem<String>(enabled: false, child: Text('自動播放語音')),
        PopupMenuItem<String>(enabled: false, child: Text('協助增加資料庫')),
      ],
      child: const _ContributionCircleButton(
        tooltip: '開啟選單',
        icon: Icons.menu_rounded,
      ),
    );
  }

  Widget _buildBottomPanel() {
    return Material(
      color: const Color(0xFFF0F0F0),
      elevation: 7,
      borderRadius: BorderRadius.circular(28),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 7, 14, 14),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 52,
              height: 5,
              decoration: BoxDecoration(
                color: Colors.black38,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            const SizedBox(height: 9),
            SizedBox(
              height: 64,
              child: TextField(
                key: const Key('contribution-text-input'),
                controller: _wordController,
                enabled: !_isRecording,
                maxLines: 1,
                textAlignVertical: TextAlignVertical.center,
                decoration: InputDecoration(
                  hintText: '輸入意思…',
                  filled: true,
                  fillColor: Colors.white,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(22),
                    borderSide: BorderSide.none,
                  ),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 18),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _ContributionCircleButton(
                  tooltip: '切換鏡頭',
                  icon: Icons.cameraswitch_outlined,
                  onPressed: _cameras.length > 1 && !_isRecording
                      ? _switchCamera
                      : null,
                ),
                _buildRecordButton(),
                _ContributionCircleButton(
                  key: const Key('contribution-upload-button'),
                  tooltip: '上傳',
                  icon: Icons.file_upload_outlined,
                  onPressed: _isRecording ? null : _prepareUpload,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    final player = _videoPlayerController;
    if (_recordedVideo != null &&
        player != null &&
        player.value.isInitialized) {
      return Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: player.value.aspectRatio,
              child: VideoPlayer(player),
            ),
          ),
          Center(
            child: IconButton.filledTonal(
              key: const Key('contribution-video-play-button'),
              tooltip: player.value.isPlaying ? '暫停預覽' : '播放預覽',
              iconSize: 38,
              onPressed: () {
                setState(() {
                  player.value.isPlaying ? player.pause() : player.play();
                });
              },
              icon: Icon(
                player.value.isPlaying
                    ? Icons.pause_rounded
                    : Icons.play_arrow_rounded,
              ),
            ),
          ),
        ],
      );
    }
    final controller = _cameraController;
    if (_isInitializing) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }
    if (_cameraError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.videocam_off_outlined,
                color: Colors.white70,
                size: 64,
              ),
              const SizedBox(height: 12),
              Text(
                _cameraError!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white),
              ),
              const SizedBox(height: 12),
              FilledButton(
                onPressed: _initializeCamera,
                child: const Text('重試'),
              ),
            ],
          ),
        ),
      );
    }
    if (controller == null || !controller.value.isInitialized) {
      return const Center(
        child: Icon(
          Icons.videocam_off_outlined,
          color: Colors.white70,
          size: 64,
        ),
      );
    }
    final previewSize = controller.value.previewSize;
    if (previewSize == null) return const SizedBox.expand();
    return ClipRect(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: previewSize.height,
          height: previewSize.width,
          child: CameraPreview(controller),
        ),
      ),
    );
  }

  Widget _buildRecordButton() {
    return InkWell(
      customBorder: const CircleBorder(),
      onTap: _isInitializing ? null : _toggleRecording,
      child: Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          color: Colors.white,
          shape: BoxShape.circle,
          border: Border.all(color: Colors.black26),
          boxShadow: const [
            BoxShadow(
              color: Colors.black26,
              blurRadius: 6,
              offset: Offset(0, 2),
            ),
          ],
        ),
        alignment: Alignment.center,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          width: _isRecording ? 22 : 44,
          height: _isRecording ? 22 : 44,
          decoration: BoxDecoration(
            color: Colors.red,
            borderRadius: BorderRadius.circular(_isRecording ? 5 : 22),
          ),
        ),
      ),
    );
  }
}

class _ContributionCircleButton extends StatelessWidget {
  const _ContributionCircleButton({
    super.key,
    required this.tooltip,
    required this.icon,
    this.onPressed,
  });

  final String tooltip;
  final IconData icon;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final button = Material(
      color: Colors.white.withValues(alpha: 0.92),
      shape: const CircleBorder(),
      elevation: 3,
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onPressed,
        child: SizedBox(
          width: 52,
          height: 52,
          child: Icon(
            icon,
            size: 29,
            color: onPressed == null ? Colors.black45 : Colors.black87,
          ),
        ),
      ),
    );
    return Tooltip(message: tooltip, child: button);
  }
}
