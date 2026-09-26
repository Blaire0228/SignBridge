import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/main.dart';

void main() {
  testWidgets('shows the translator home page', (WidgetTester tester) async {
    await tester.pumpWidget(const TslTranslatorApp(enableModelViewer: false));

    expect(find.text('TSL'), findsNothing);
    expect(find.text('1.0x'), findsOneWidget);
    expect(find.byIcon(Icons.replay_rounded), findsOneWidget);

    final speechButton = find.byKey(const Key('speech-input-button'));
    final sendButton = find.byKey(const Key('text-send-button'));
    expect(speechButton, findsOneWidget);
    expect(sendButton, findsOneWidget);
    expect(
      tester.getCenter(speechButton).dx,
      lessThan(tester.getCenter(sendButton).dx),
    );
    final inputField = find.byType(TextField);
    expect(
      tester.getCenter(speechButton).dy,
      closeTo(tester.getCenter(inputField).dy, 1),
    );
    expect(
      tester.getCenter(sendButton).dy,
      closeTo(tester.getCenter(inputField).dy, 1),
    );

    tester.view.viewInsets = const FakeViewPadding(bottom: 300);
    await tester.pump();
    tester.view.resetViewInsets();
    await tester.pump();

    await tester.tap(find.byKey(const Key('playback-speed-button')));
    await tester.pump();
    expect(find.text('1.5x'), findsOneWidget);

    await tester.tap(find.byKey(const Key('playback-speed-button')));
    await tester.pump();
    expect(find.text('0.5x'), findsOneWidget);

    await tester.tap(find.byKey(const Key('replay-button')));
    await tester.pump();

    expect(find.byKey(const Key('input-sheet-drag-handle')), findsOneWidget);
    await tester.drag(
      find.byKey(const Key('input-sheet-drag-handle')),
      const Offset(0, -180),
    );
    await tester.pump(const Duration(milliseconds: 500));

    await tester.tap(find.byKey(const Key('home-menu-button')));
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.text('自動播放動畫'), findsOneWidget);
    expect(find.text('自動播放語音'), findsOneWidget);
    expect(find.text('協助增加資料庫'), findsOneWidget);
    expect(find.text('語言切換?'), findsNothing);
    expect(find.textContaining('登入'), findsNothing);

    expect(find.text('偵測語言'), findsOneWidget);
    expect(find.text('台灣手語'), findsOneWidget);
    await tester.tapAt(const Offset(100, 500));
    await tester.pumpAndSettle();

    final fixedChatButton = find.byKey(const Key('fixed-chat-button'));
    expect(fixedChatButton, findsOneWidget);
    expect(tester.getTopLeft(fixedChatButton), const Offset(12, 12));
    expect(tester.getSize(fixedChatButton), const Size(44, 44));
    await tester.tap(fixedChatButton);
    await tester.pumpAndSettle();
    expect(find.text('選擇對話方式'), findsOneWidget);
    expect(find.text('建立臨時房間'), findsOneWidget);
    expect(find.text('掃描 QR Code'), findsOneWidget);
    expect(find.text('句子辨識'), findsNothing);
  });
}
